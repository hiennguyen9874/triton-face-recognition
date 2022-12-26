import asyncio
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


def batching_list(iterable, batch_size: int = 1):
    len_iterable = len(iterable)
    for first_idx in range(0, len_iterable, batch_size):
        yield iterable[first_idx : min(first_idx + batch_size, len_iterable)]


def output_tensor_to_numpy(output_tensor):
    return (
        output_tensor.as_numpy()
        if output_tensor.is_cpu()
        else from_dlpack(output_tensor.to_dlpack()).detach().cpu().numpy()
    )


def crop(image, box, lmks):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    new_image = image[y1:y2, x1:x2, :].copy()
    lmks = lmks.copy()
    lmks[0::2] -= x1
    lmks[1::2] -= y1
    return new_image, lmks


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def get_environ_or_error(name: str, default=None):
        value = os.environ.get(name, default)
        if value is None:
            raise pb_utils.TritonModelException(f"{name} not set")
        return value

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

        self.face_detection_batch_size = int(
            self.get_environ_or_error("FACE_DETECTION_BATCH_SIZE", 4)
        )
        self.face_recogition_batch_size = int(
            self.get_environ_or_error("FACE_RECOGITION_BATCH_SIZE", 16)
        )
        self.face_recogition_output_dim = int(
            self.get_environ_or_error("FACE_RECOGITION_OUTPUT_DIM", 512)
        )

    async def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameterswidth, height ratios
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            image_data = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)

            # Face detection preprocessing

            inference_response_awaits = []
            for image in image_data:
                image = np.expand_dims(image, axis=0)

                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name="FaceDetectionPreprocessing",
                    requested_output_names=["IMAGE", "IMGSIZE", "RATIO", "PAD"],
                    inputs=[
                        pb_utils.Tensor("IMAGE", image.astype(np.uint8)),
                    ],
                )
                inference_response_awaits.append(infer_request.async_exec())

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(*inference_response_awaits)

            images = []
            imgsizes = []
            ratios = []
            pads = []

            for infer_response in inference_responses:
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                images.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "IMAGE")
                    )
                )
                imgsizes.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "IMGSIZE")
                    )
                )
                ratios.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "RATIO")
                    )
                )
                pads.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "PAD")
                    )
                )

            images = np.concatenate(images, axis=0)
            imgsizes = np.concatenate(imgsizes, axis=0)
            ratios = np.concatenate(ratios, axis=0)
            pads = np.concatenate(pads, axis=0)

            # Face detection

            inference_response_awaits = []
            for batch in batching_list(images, self.face_detection_batch_size):
                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name="FaceDetection",
                    requested_output_names=[
                        "num_dets",
                        "det_boxes",
                        "det_scores",
                        "det_classes",
                        "det_lmks",
                    ],
                    inputs=[
                        pb_utils.Tensor("images", np.stack(batch, axis=0).astype(np.float32)),
                    ],
                )
                inference_response_awaits.append(infer_request.async_exec())

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(*inference_response_awaits)

            num_dets = []
            det_boxes = []
            det_scores = []
            det_classes = []
            det_lmks = []

            for infer_response in inference_responses:
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                num_dets.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "num_dets")
                    )
                )

                det_boxes.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_boxes")
                    )
                )
                det_scores.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_scores")
                    )
                )
                det_classes.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_classes")
                    )
                )
                det_lmks.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_lmks")
                    )
                )

            num_dets = np.concatenate(num_dets, axis=0)
            det_boxes = np.concatenate(det_boxes, axis=0)
            det_scores = np.concatenate(det_scores, axis=0)
            det_classes = np.concatenate(det_classes, axis=0)
            det_lmks = np.concatenate(det_lmks, axis=0)

            # Face detection postprocessing

            inference_response_awaits = []
            for batch in batching_list(
                list(
                    zip(
                        det_boxes,
                        det_lmks,
                        imgsizes,
                        ratios,
                        pads,
                    )
                ),
                self.face_detection_batch_size,
            ):
                batch = list(zip(*batch))

                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name="FaceDetectionPostProcessing",
                    requested_output_names=["det_boxes", "det_lmks"],
                    inputs=[
                        pb_utils.Tensor("det_boxes", np.stack(batch[0], axis=0).astype(np.float32)),
                        pb_utils.Tensor("det_lmks", np.stack(batch[1], axis=0).astype(np.float32)),
                        pb_utils.Tensor("IMGSIZE", np.stack(batch[2], axis=0).astype(np.int32)),
                        pb_utils.Tensor("RATIO", np.stack(batch[3], axis=0).astype(np.float32)),
                        pb_utils.Tensor("PAD", np.stack(batch[4], axis=0).astype(np.float32)),
                    ],
                )
                inference_response_awaits.append(infer_request.async_exec())

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(*inference_response_awaits)

            det_boxes = []
            det_lmks = []

            for infer_response in inference_responses:
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                det_boxes.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_boxes")
                    )
                )
                det_lmks.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "det_lmks")
                    )
                )

            det_boxes = np.concatenate(det_boxes, axis=0)
            det_lmks = np.concatenate(det_lmks, axis=0)

            # Face recogition preprocessing
            batch_idxs = []
            box_idxs = []

            inference_response_awaits = []
            for batch_idx, (num_det, det_box, det_score, det_classe, det_lmk) in enumerate(
                zip(num_dets, det_boxes, det_scores, det_classes, det_lmks)
            ):
                det_box = det_box[: num_det[0]]
                det_score = det_score[: num_det[0]]
                det_classe = det_classe[: num_det[0]]
                det_lmk = det_lmk[: num_det[0]]

                for box_idx, (box, score, cls, lmk) in enumerate(
                    zip(det_box, det_score, det_classe, det_lmk)
                ):
                    if score < 0.25 or cls not in [0]:
                        continue

                    img, lmk = crop(image_data[batch_idx], box, lmk)

                    batch_idxs.append(batch_idx)
                    box_idxs.append(box_idx)

                    # Create inference request object
                    infer_request = pb_utils.InferenceRequest(
                        model_name="FaceRecognitionPreprocessing",
                        requested_output_names=["OUTPUT"],
                        inputs=[
                            pb_utils.Tensor("INPUT0", np.expand_dims(img, axis=0).astype(np.uint8)),
                            pb_utils.Tensor(
                                "INPUT1", np.expand_dims(lmk, axis=0).astype(np.float32)
                            ),
                        ],
                    )
                    inference_response_awaits.append(infer_request.async_exec())

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(*inference_response_awaits)

            images = []
            for infer_response in inference_responses:
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                images.append(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT")
                    )
                )

            images = np.concatenate(images, axis=0)

            # Face recogition
            batch_idxs2 = []
            box_idxs2 = []

            inference_response_awaits = []
            for batch in batching_list(
                list(zip(images, batch_idxs, box_idxs)), batch_size=self.face_recogition_batch_size
            ):
                batch = list(zip(*batch))

                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name="FaceRecognition",
                    requested_output_names=["683"],
                    inputs=[
                        pb_utils.Tensor("input.1", np.stack(batch[0], axis=0).astype(np.float32))
                    ],
                )
                inference_response_awaits.append(infer_request.async_exec())
                batch_idxs2.append(batch[1])
                box_idxs2.append(batch[2])

            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            inference_responses = await asyncio.gather(*inference_response_awaits)

            has_features = np.zeros(
                (
                    det_boxes.shape[0],
                    det_boxes.shape[1],
                ),
                dtype=np.bool,
            )
            features = np.zeros(
                (det_boxes.shape[0], det_boxes.shape[1], self.face_recogition_output_dim),
                dtype=np.float32,
            )

            for infer_response, batch_idx2, box_idx2 in zip(
                inference_responses, batch_idxs2, box_idxs2
            ):
                # Make sure that the inference response doesn't have an error.
                # If it has an error and you can't proceed with your model
                # execution you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                for feature, batch_idx, box_idx in zip(
                    output_tensor_to_numpy(
                        pb_utils.get_output_tensor_by_name(infer_response, "683")
                    ),
                    batch_idx2,
                    box_idx2,
                ):
                    has_features[batch_idx, box_idx] = 1
                    features[batch_idx, box_idx] = feature

            # Create reponse
            num_dets = pb_utils.Tensor("num_dets", num_dets.astype(np.int32))
            det_boxes = pb_utils.Tensor("det_boxes", det_boxes.astype(np.float32))
            det_scores = pb_utils.Tensor("det_scores", det_scores.astype(np.float32))
            det_classes = pb_utils.Tensor("det_classes", det_classes.astype(np.int32))
            det_lmks = pb_utils.Tensor("det_lmks", det_lmks.astype(np.float32))
            has_features = pb_utils.Tensor("has_features", has_features.astype(np.bool))
            features = pb_utils.Tensor("features", features.astype(np.float32))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    num_dets,
                    det_boxes,
                    det_scores,
                    det_classes,
                    det_lmks,
                    has_features,
                    features,
                ]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")

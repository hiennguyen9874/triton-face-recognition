import asyncio
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


def face_recogition_preprocessing(image, lmks, target_shape):
    origin_shape = image.shape[:2]
    image = cv2.resize(image, target_shape)
    lmks[0::2] *= target_shape[1] / origin_shape[1]
    lmks[1::2] *= target_shape[0] / origin_shape[0]
    return image, lmks


def ds_preprocess(
    image,
    net_scale_factor: float = 0.0039215697906911373,
    offsets: Optional[List[float]] = None,
):
    if offsets is None:
        offsets = [0, 0, 0]
    image = net_scale_factor * (image - np.array(offsets))
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.ascontiguousarray(image)  # contiguous
    image = image.astype(np.float32)
    return image


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

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.output0_dims = output0_config["dims"]

        self.face_aligment_batch_size = int(
            self.get_environ_or_error("FACE_ALIGMENT_BATCH_SIZE", 256)
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
            image_data = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)

            # Get INPUT1
            lmk_data = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()
            if image_data.ndim == 1:
                image_data = np.expand_dims(image_data, axis=0)

            inference_response_awaits = []
            for batch in batching_list(
                list(zip(image_data, lmk_data)), self.face_aligment_batch_size
            ):
                images = []
                lmks = []
                for image, lmk in batch:
                    image, lmk = face_recogition_preprocessing(
                        image,
                        lmk,
                        (self.output0_dims[1], self.output0_dims[2]),
                    )
                    images.append(image)
                    lmks.append(lmk)

                if images and lmks:
                    images = np.stack(images, axis=0)
                    lmks = np.stack(lmks, axis=0)

                    # Create inference request object
                    infer_request = pb_utils.InferenceRequest(
                        model_name="FaceAligment",
                        requested_output_names=["OUTPUT"],
                        inputs=[
                            pb_utils.Tensor("INPUT0", images.astype(np.uint8)),
                            pb_utils.Tensor("INPUT1", lmks.astype(np.float32)),
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

                for image in output_tensor_to_numpy(
                    pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT")
                ):
                    image = ds_preprocess(
                        image, net_scale_factor=0.00784313725, offsets=[127.5, 127.5, 127.5]
                    )
                    images.append(image)

            assert len(images) == image_data.shape[0] == lmk_data.shape[0]

            images = np.stack(images, axis=0)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT", images.astype(np.float32)),
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

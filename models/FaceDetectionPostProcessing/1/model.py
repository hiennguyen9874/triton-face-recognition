import json

import numpy as np
import triton_python_backend_utils as pb_utils


def yolo_postprocessing(det_boxes, det_lmks, ratio, pad, img_size):
    det_boxes = det_boxes.copy()
    det_lmks = det_lmks.copy()

    det_boxes[..., 0::2] = np.clip((det_boxes[..., 0::2] - pad[0]) / ratio[0], 0, img_size[0])
    det_boxes[..., 1::2] = np.clip((det_boxes[..., 1::2] - pad[1]) / ratio[1], 0, img_size[1])

    det_lmks[..., 0::2] = np.clip((det_lmks[..., 0::2] - pad[0]) / ratio[0], 0, img_size[0])
    det_lmks[..., 1::2] = np.clip((det_lmks[..., 1::2] - pad[1]) / ratio[1], 0, img_size[1])

    return det_boxes, det_lmks


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

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

    def execute(self, requests):
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

            out_boxes = []
            out_lmks = []

            for (det_boxes, det_lmks, imgsize, ratio, pad) in zip(
                pb_utils.get_input_tensor_by_name(request, "det_boxes").as_numpy(),
                pb_utils.get_input_tensor_by_name(request, "det_lmks").as_numpy(),
                pb_utils.get_input_tensor_by_name(request, "IMGSIZE").as_numpy(),
                pb_utils.get_input_tensor_by_name(request, "RATIO").as_numpy(),
                pb_utils.get_input_tensor_by_name(request, "PAD").as_numpy(),
            ):
                out_box, out_lmk = yolo_postprocessing(
                    det_boxes, det_lmks, ratio, pad, imgsize[::-1]
                )
                out_boxes.append(out_box)
                out_lmks.append(out_lmk)

            out_boxes = np.stack(out_boxes, axis=0)
            out_lmks = np.stack(out_lmks, axis=0)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("det_boxes", out_boxes.astype(np.float32)),
                    pb_utils.Tensor("det_lmks", out_lmks.astype(np.float32)),
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

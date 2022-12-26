import json
from typing import List, Optional

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


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

        # Get IMAGE configuration
        image_config = pb_utils.get_output_config_by_name(self.model_config, "IMAGE")
        self.image_dims = image_config["dims"]

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
            image_data = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()

            img_out = []
            imgsize_out = []
            ratio_out = []
            pad_out = []
            for image in image_data if image_data.ndim == 4 else [image_data]:
                imgsize_out.append(image.shape[:2])
                image, ratio, pad = letterbox(
                    image, new_shape=(self.image_dims[1], self.image_dims[1]), auto=False, stride=32
                )
                image = ds_preprocess(
                    image, net_scale_factor=0.0039215697906911373, offsets=[0, 0, 0]
                )
                img_out.append(image)
                ratio_out.append(ratio)
                pad_out.append(pad)
            img_out = np.stack(img_out, axis=0)
            imgsize_out = np.array(imgsize_out, dtype=np.int32)
            ratio_out = np.array(ratio_out, dtype=np.float32)
            pad_out = np.array(pad_out, dtype=np.float32)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("IMAGE", img_out.astype(np.float32)),
                    pb_utils.Tensor("IMGSIZE", imgsize_out.astype(np.int32)),
                    pb_utils.Tensor("RATIO", ratio_out.astype(np.float32)),
                    pb_utils.Tensor("PAD", pad_out.astype(np.float32)),
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

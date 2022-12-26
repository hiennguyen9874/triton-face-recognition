import json

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from skimage import transform as trans


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

        # Get OUTPUT configuration
        output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.output_dims = output_config["dims"]

        self.output_size = (self.output_dims[1], self.output_dims[0])

        self.default_array = np.array(
            [
                [30.29459953 + 8, 51.69630051],
                [65.53179932 + 8, 51.50139999],
                [48.02519989 + 8, 71.73660278],
                [33.54930115 + 8, 92.3655014],
                [62.72990036 + 8, 92.20410156],
            ],
            dtype=np.float32,
        )

    def align_image(self, image, landmarks):
        landmarks = np.array(landmarks, dtype=np.float32).reshape(5, 2)
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.default_array)
        tfm = tform.params[0:2, :]
        return cv2.warpAffine(image, tfm, (self.output_size[0], self.output_size[1]))

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output_dtype = self.output_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.

        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()

            # Get INPUT1
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()

            aligned_image = np.stack(
                [self.align_image(img, lmk) for img, lmk in zip(in_0, in_1)], axis=0
            )

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor = pb_utils.Tensor("OUTPUT", aligned_image.astype(output_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
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

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger()

CLASSES = ('_background', 'Roads')

model_input_w = 256
model_input_h = 128

color_list = [[0, 0, 0], [255, 0, 0]]



# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def img_preprocess(src_img):
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_input_w, model_input_h))

    img = img.astype(np.float32)
    img = img * 0.003921568
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img
    

def main():
    engine_file_path = './DeepLabV3.trt'
    input_image_path = './test.jpg'

    orig_img = cv2.imread(input_image_path)
    img_h, img_w = orig_img.shape[:2]
    image = img_preprocess(orig_img)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = image
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

    output = trt_outputs[0]

    out_b, out_c, out_h, out_w = 1, len(CLASSES), model_input_h, model_input_w

    mask = np.zeros(shape=(out_h, out_w, 3))
    for h in range(out_h):
        for w in range(out_w):
            maxValue = output[0 * out_h * out_w + h * out_w + w]
            maxIndex = 0
            for c in range(1, out_c, 1):
                if  maxValue < output[c * out_h * out_w + h * out_w + w]:
                    maxValue = output[c * out_h * out_w + h * out_w + w]
                    maxIndex = c
            mask[h, w, :] = color_list[maxIndex]

    mask = cv2.resize(mask, (img_w, img_h))
    opencv_image = np.clip(np.array(orig_img)*0.8 + np.array(mask)*0.55, a_min=0, a_max=255)
    opencv_image = opencv_image.astype("uint8")
    cv2.imwrite('./test_result_tensorRT.jpg', opencv_image)
    print('Finished!')


if __name__ == '__main__':
    print('This is main ...')
    main()

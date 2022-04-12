import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ('_background', 'Roads')

model_input_w = 256
model_input_h = 128

color_list = [[0, 0, 0], [255, 0, 0]]


def img_preprocess(src_img):
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_input_w, model_input_h))

    img = img.astype(np.float32)
    img = img * 0.003921568
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


def test_image(img_path, onnx_path):
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]
    img_data = img_preprocess(orig_img)
    img_data = np.expand_dims(img_data, axis=0)

    ort_session = ort.InferenceSession(onnx_path)
    out = ort_session.run(None, {'data': img_data})

    output = out[0].reshape((-1))

    out_b, out_c, out_h, out_w = 1, len(CLASSES), model_input_h, model_input_w

    mask = np.zeros(shape=(out_h, out_w, 3))
    for h in range(out_h):
        for w in range(out_w):
            maxValue = output[0 * out_h * out_w + h * out_w + w]
            maxIndex = 0
            for c in range(1, out_c, 1):
                if maxValue < output[c * out_h * out_w + h * out_w + w]:
                    maxValue = output[c * out_h * out_w + h * out_w + w]
                    maxIndex = c
            mask[h, w, :] = color_list[maxIndex]

    mask = cv2.resize(mask, (img_w, img_h))
    opencv_image = np.clip(np.array(orig_img) * 0.8 + np.array(mask) * 0.55, a_min=0, a_max=255)
    opencv_image = opencv_image.astype("uint8")
    cv2.imwrite('./test1_result_onnx.jpg', opencv_image)
    print('Finished!')


if __name__ == "__main__":
    print('This is main ...')
    img_path = './test1.jpg'
    onnx_path = './DeepLabV3.onnx'
    test_image(img_path, onnx_path)
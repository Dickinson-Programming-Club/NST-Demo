import io
import time
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import logging
from tqdm import tqdm  

# Initializing a basic logger. Shouldn't really be needed for this application, but it's good to have.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("style_transfer_app")

# Set inter and intra op parallelism options to 4.
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

app = FastAPI(title="Style Transfer API", description="API for performing neural style transfer on images.")

def load_img(img_bytes):
    max_dim = 256 # I would like to scale this to 512x512 but the performance struggles real bad on free hardware. Either I scale up to 512 and reduce the epochs and training steps by half or keep the dim at 256 while keeping image quality.
    img = tf.image.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    loss = style_loss + content_loss
    return loss

@tf.function
def train_step(image, extractor, style_targets, content_targets, opt, style_weight, content_weight, total_variation_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)  
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']
extractor = StyleContentModel(style_layers, content_layers)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...)
):
    try:
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()
        
        content_img = load_img(content_bytes)
        style_img = load_img(style_bytes)

        style_targets = extractor(style_img)['style']
        content_targets = extractor(content_img)['content']
        image = tf.Variable(content_img)

        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        style_weight = 1e-2
        content_weight = 1e4
        total_variation_weight = 30

        epochs = 10
        steps_per_epoch = 100

        start_time = time.time()
        losses = []

        for n in tqdm(range(epochs), desc="Epochs"): 
            for m in tqdm(range(steps_per_epoch), desc="Steps", leave=False):  
                loss = train_step(image, extractor, style_targets, content_targets, opt, style_weight, content_weight, total_variation_weight)
                losses.append(float(loss.numpy()))  

        end_time = time.time()
        total_time = end_time - start_time

        result = tensor_to_image(image)

        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')

        response = {
            "image": encoded_image,
            "statistics": {
                "total_time": total_time,
                "losses": losses,
                "style_weight": style_weight,
                "content_weight": content_weight,
                "total_variation_weight": total_variation_weight,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch
            }
        }

        return JSONResponse(content=response)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during style transfer: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during style transfer.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

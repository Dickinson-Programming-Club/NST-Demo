import requests
import base64

def test_style_transfer():
    url = "http://localhost:8000/style-transfer"
    content_image_path = "images/test_base_image.jpg"
    style_image_path = "images/test_style_image.jpg"

    print(f"Attempting to transfer style from {style_image_path} to {content_image_path}...")

    with open(content_image_path, "rb") as content_image, open(style_image_path, "rb") as style_image:
        files = {
            "content_image": content_image,
            "style_image": style_image
        }
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Style transfer successful!")
        result = response.json()
        
        # Decode the Base64 string to bytes
        image_data = base64.b64decode(result["image"])

        with open("result_image.png", "wb") as f:
            f.write(image_data)
        print("Result image saved as result_image.png")
        print("Statistics:", result["statistics"])
    else:
        print(f"Style transfer failed with status code: {response.status_code}")
        print("Error message:", response.text)

if __name__ == "__main__":
    test_style_transfer()
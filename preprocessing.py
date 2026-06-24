import cv2
import numpy as np

def cropper(image):
  """
  since the image is coming as an object from PIL and not from numpy, 
  we invoke the size attribute (PIL library) instead of shape attributes

  Args:
      image: image opened using Image.open from PIL, so it's a PIL object

  Returns:
      cropped square image    
  """
  width, height = image.size
  mindim = min(width, height)

  left = (width - mindim) // 2
  top = (height - mindim) // 2

  cropped_img = image.crop((left, top, left + mindim, top + mindim))

  return cropped_img


def clahe(green_channel, clip_limit=2.0, grid_size=(8,8)):
    """
    Applies CLAHE to enhance local contrast on a single grayscale channel.
    
    Args:
        green_channel: numpy array shape (H, W), dtype uint8
        clip_limit: contrast clipping threshold
        grid_size: tile grid size for local histogram
    
    Returns:
        enhanced_channel: numpy array shape (H, W), dtype uint8
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    enhanced = clahe.apply(green_channel)
    return enhanced

def build_color_channels(img_array, green_enhanced):
    """
    img_array: numpy array shape (H, W, 3) in RGB order, dtype uint8
    green_enhanced: numpy array shape (H, W), CLAHE-applied green channel, dtype uint8
    
    Returns: numpy array shape (H, W, 3), dtype uint8
    """
    # Extract original channels
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    
    # Warm filter: (R - B) clipped to [0, 255]
    warm = np.clip(r - b, 0, 255).astype(np.uint8)
    
    # Green channel: use the CLAHE-enhanced version
    green = green_enhanced  # already uint8
    
    # Grayscale: standard luminance formula 0.299*R + 0.587*G + 0.114*B
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    
    # Stack into final 3-channel image
    result = np.stack([warm, green, gray], axis=-1)  # shape (H, W, 3)
    return result

def resize_image(img_array,size=224):
   resized_image=cv2.resize(img_array,(size,size),interpolation=cv2.INTER_LINEAR)

   return resized_image
  
def pipeline(pil_image, target_size=224):
    
    cropped = cropper(pil_image)
    
    
    img_array = np.array(cropped)  # (H, W, 3), uint8, RGB
    
    green_raw = img_array[:, :, 1]  # index 1 = green in RGB
    green_enhanced = clahe(green_raw)
    
    custom_img = build_color_channels(img_array, green_enhanced)
    
    resized = resize_image(custom_img, target_size)
    
    return resized  # numpy array (224, 224, 3), uint8

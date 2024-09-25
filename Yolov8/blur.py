import numpy as np
from PIL import Image, ImageFilter

def apply_gaussian_blur(image, radius=2):
    """
    이미지에 Gaussian 블러를 적용하는 함수
    :param image: 입력 이미지 (PIL 이미지 객체)
    :param radius: 블러 강도 (값이 클수록 더 흐려짐)
    :return: Gaussian 블러가 적용된 이미지 (PIL 이미지 객체)
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    이미지에 Gaussian 노이즈를 추가하는 함수
    :param image: 입력 이미지 (PIL 이미지 객체)
    :param mean: 노이즈의 평균값
    :param sigma: 노이즈의 표준편차 (값이 클수록 노이즈가 커짐)
    :return: 노이즈가 추가된 이미지 (PIL 이미지 객체)
    """
    img_array = np.array(image.convert("L"))
    gaussian_noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def process_image_with_blur_and_noise(image_path, blur_radius, noise_sigma):
    """
    Gaussian 블러와 Gaussian 노이즈를 동시에 적용하는 함수
    :param image_path: 이미지 경로
    :param blur_radius: Gaussian 블러 강도
    :param noise_sigma: Gaussian 노이즈 강도 (sigma 값)
    :return: 처리된 이미지 (PIL 이미지 객체)
    """
    # 이미지 불러오기
    image = Image.open(image_path)

    # Gaussian 블러 적용
    blurred_image = apply_gaussian_blur(image, radius=blur_radius)

    # Gaussian 노이즈 추가
    noisy_image = add_gaussian_noise(blurred_image, sigma=noise_sigma)

    return noisy_image
import os
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Firefox()
driver.get("http://localhost:3000/")

personal_images = driver.find_elements(By.TAG_NAME, "img")

image_counter = 1
for image in personal_images:
    urllib.request.urlretrieve(image.get_attribute('src'), os.path.join("personal", f"personal_{image_counter}.jpg"))
    image_counter += 1
#     with open(os.path.join("personal", f"personal_{image_counter}.png"), "wb") as file:
#         file.write(image)
#     image_counter += 1

visited_urls = []

go_to_users = driver.find_element(By.TAG_NAME, "a")

go_to_users.click()

profiles = driver.find_elements(By.TAG_NAME, "a")
image_counter = 1
for i in range(len(profiles)):
    profiles = driver.find_elements(By.TAG_NAME, "a")
    profiles[i].click()
    visited_urls.append(driver.current_url)
    others_images = driver.find_elements(By.TAG_NAME, "img")
    
    for img in others_images:
        urllib.request.urlretrieve(img.get_attribute('src'), os.path.join("others", f"others_{image_counter}.jpg"))
        image_counter += 1

    back_to_users = driver.find_element(By.TAG_NAME, 'a')
    back_to_users.click()
    
    
print(visited_urls)
driver.close()

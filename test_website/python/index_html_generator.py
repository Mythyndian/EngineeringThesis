import os


personal_images = os.listdir(os.path.join("images", "personal"))
print(personal_images)
head = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Gallery</title>
    <meta charset="utf-8" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="style.css">
  </head>
  <body>
    <div class="container">
      <h2>Username_placeholder</h2>
      <p>Biography_placeholder</p>
      <h2>Contact info</h2>
      <p>email_placeholder</p>
    </div>
    <div class="container">
"""

images_html = ""

for image in personal_images:
    images_html += f"""
    <img src="{os.path.join('personal',image)}"  alt="..." class="img-fluid border-dark personal">
    """

tail = """
    </div>
    <div class="container">
      <a class="btn btn-primary" href="users.html" role="button">See more users</a>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
    </body>
</html>
"""

html_file = head + images_html + tail

with open(os.path.join("html", "index.html"), "w", encoding="utf-8") as file:
    file.write(html_file)

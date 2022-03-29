import os
import numpy as np


class UserProfile:
    def __init__(self, images_per_profile: int, image_list) -> None:
        self.image_names = image_list
        self.image_limit = images_per_profile
        self.splits = np.array_split(image_list, len(image_list) // images_per_profile)
        self.header = """
        <!DOCTYPE html>
<html lang="en">
  <head>
    <title>Gallery</title>
    <meta charset="utf-8" />
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC"
      crossorigin="anonymous"
    />
    <link rel="stylesheet" type="text/css" href="style.css" />
  </head>
        
        <body>
    <div class="container">
      <h2>User</h2>
      <p>Biography_placeholder</p>
      <h2>Contact info</h2>
      <p>email_placeholder</p>
    </div>
        """

    def generate_file(self):
        start = """<div class="container">
      <h3>User Gallery</h3>"""
        end = """<div class="container">
      <a class="btn btn-primary" href="users.html" role="button">See more users</a>
    </div>

    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM"
      crossorigin="anonymous"
    ></script>
  </body>
</html>"""
        counter = 0
        middle = """"""
        for imgs in self.splits:
            counter += 1
            middle = """"""
            for img_path in imgs:
                middle += f"""    <img
        src="others\{img_path}"
        alt="..."
        class="img-fluid border-dark personal"
      />"""
            with open(f"html/user_{counter}.html", "w") as html_file:
                html_file.write(self.header + start + middle + end)

p = UserProfile(5, os.listdir(os.path.join("images","others")))
p.generate_file()
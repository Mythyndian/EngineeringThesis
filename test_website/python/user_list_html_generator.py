import os

# read for generated user profiles htmls
user_profiles = os.listdir(os.path.join("html"))
user_profiles.remove("index.html")
user_profiles.remove("users.html")

header = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC"
      crossorigin="anonymous"
    />
    <link rel="stylesheet" type="text/css" href="style.css" />
    <title>Document</title>
  </head>
  <body>
    <div class="container">
      <h2>Site users</h2>
"""

cards_html = ""

for user_profile in user_profiles:
    cards_html += f"""
    <div class="card" style="width: 18rem">
        <div class="card-body">
          <h5 class="card-title">{user_profile.replace('.html', '')}</h5>
          <p class="card-text">
            Some quick example text to build on the card title and make up the
            bulk of the card's content.
          </p>
          <a href="{user_profile}" class="btn btn-primary">Go somewhere</a>
        </div>
      </div>
    """

footer = """
</div>

    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM"
      crossorigin="anonymous"
    ></script>
  </body>
</html>
"""

with open(os.path.join("html", "users.html"), "w") as file:
    file.write(header + cards_html + footer)

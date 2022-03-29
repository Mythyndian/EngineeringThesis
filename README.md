# Implementacja metody szybkiego wykrywania duplikatów

## Konfiguracja środowiska uruchomieniowego Python
### W pierwszej kolejności należy utworzyć za pomocą narzędzia venv lub dowolnego innego służącego do tworzenia środowisk virtualnych.
Windows: `python -m venv env`
Linux: `python3 -m venv env`
### Efektem takiego polecenia będzie utworzenie folderu env z plikami środowiska wirtualnego.

### Następny krokiem jest aktywacja środowiska następującymi poleceniami
`cd  env/Scripts`
Windows: `.\activate`
Linux: `source env/bin/activate`

### Instalacja wymaganych pakietów
Windows: `pip install -r requirements.txt`
Linux: `pip3 install -r requirements.txt`

### Konfiguracja środowiska uruchomieniowego Node.js
Do poprawnego działania serwera lokalnego jest wymagana instalacja [Node.js](https://nodejs.org/en/) 
oraz biblioteki [Express.js](https://expressjs.com/).

## Przygotowanie do uruchomienia
- Umieścić pliki graficzne w folderze EngineeringThesis/test_website/images/personal
- Z folderu EngineeringThesis/test_website/python uruchomić polecenie `python image_manipulator.py`
- Z folderu EngineeringThesis/test_website/python uruchomić polecenie `python index_html_generator.py`
- Z folderu EngineeringThesis/test_website/python uruchomić polecenie `python user_profile_generator.py`
- Z folderu EngineeringThesis/test_website/python uruchomić polecenie `python user_list_generator.py`
- Z folderu EngineeringThesis/test_website/python uruchomić polecenie `node server.js`
## Uruchomienie web crawlera
Z folderu EngineeringThesis/algorithm uruchomić polecenie `python selenium_webcrawler.py`
## Uruchomienie algorytmu
Z folderu EngineeringThesis/algorithm `python image_matching_module.py`

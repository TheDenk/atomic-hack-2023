# Flask приложение для детектирования дефектов на трубе

# Cтавим requirements  
pip install -r requirements.txt
  
# Пример запуска  
python app.py


# Docker


*  docker build --tag flask-tube . 

*  docker run -p PORT:5000 --name flask-tube -t flask-tube:latest  

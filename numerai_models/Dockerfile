FROM python:3.6

WORKDIR /my_numerai

COPY requirements.txt /my_numerai/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl /my_numerai/bayesopt_xgbr.pkl

ENV FLASK_APP xgb_bayesopt.py

CMD ['flask', 'run', '--host=0.0.0.0']

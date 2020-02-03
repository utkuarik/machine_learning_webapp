
FROM python:3.6
EXPOSE 8501
WORKDIR /app
ADD . ./
COPY src/requirements.txt ./
# RUN pip install streamlit
RUN pip install -r ./requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir ~/.streamlit
RUN cp streamlit/config.toml ~/.streamlit/config.toml
RUN cp streamlit/credentials.toml ~/.streamlit/credentials.toml
COPY src /app
CMD streamlit run ./fifa.py
# CMD streamlit hello

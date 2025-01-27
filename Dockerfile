FROM python:3.12
COPY . /app

# Set the working directory
WORKDIR /app

COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime. 
EXPOSE 8080
CMD ["streamlit", "run", "./streamlit/app.py", "--server.port=8080"]
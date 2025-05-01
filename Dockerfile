# Use the official Python 3.11 slim image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to keep the image size smaller
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container at /app
COPY app.py .

# Copy the .env file (Note: For production, consider injecting env vars instead of copying the file)
COPY .env .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the port (optional, but good practice)
ENV PORT 8000

# Run app.py when the container launches
# Use uvicorn to run the FastAPI application
# --host 0.0.0.0 makes the server accessible from outside the container
# --port $PORT uses the environment variable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

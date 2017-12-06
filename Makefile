CXX = nvcc
TARGET = CNNConvLayer

all: CNNConvLayer.cu
	source /opt/cuda8.sh
	$(CXX) $< -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET) 

run:
	./$(TARGET)
	
	
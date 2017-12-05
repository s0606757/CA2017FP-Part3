CXX = nvcc
TARGET1 = ip_static
TARGET2 = ip_dynamic
FLAGS = -arch=sm_35 -rdc=true -lcudadevrt


all: 
	$(CXX) ip_static.cu           $< -o $(TARGET1)
	$(CXX) ip_dynamic.cu $(FLAGS) $< -o $(TARGET2)

clean:
	rm -f $(TARGET1) $(TARGET2) 

run:
	./$(TARGET1)
	./$(TARGET2)


#nvcc test.cu ¡Vo test ¡Varch=sm_35 ¡Vrdc=true -lcudadevrt
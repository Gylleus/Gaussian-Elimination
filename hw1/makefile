SRC = pthread_gauss_elim.cpp -lpthread
OBJS = $(SRC:.cpp=.o)

CO = g++
CFLAGS = -Wall -std=c++0x
LFLAGS =
LINK_TARGET = pthread_gauss_elim openmp_gauss_elim
all : depend $(LINK_TARGET)

clean :
	rm -f $(LINK_TARGET) ./.depend

test: all
	/info/DD2387/labs/lab1/0.2_hello_world/hw_verifier $(LINK_TARGET)

depend: .depend

.depend : $(SRC)
	rm -f ./.depend
	$(CO) $(CFLAGS) -MM $^ -MF ./.depend

pthread_gauss_elim : $(OBJS)
	$(CO) $^ $(LFLAGS) -o $@

openmp_gauss_elim : 
	g++ -fopenmp -Wall openmp_gauss_elim.cpp -o openmp_gauss_elim

%.o : %.cpp
	$(CO) $< $(CFLAGS) -c

include .depend


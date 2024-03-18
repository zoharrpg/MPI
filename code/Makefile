APP_NAME=wireroute

CXX = mpic++
CXXFLAGS = -Wall -Wextra -std=c++20 -I.

OBJS := wireroute.o

all: CXXFLAGS += -O3
all: $(APP_NAME)

debug: CXXFLAGS += -O0 -g
debug: $(APP_NAME)_debug

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(APP_NAME)_debug: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) $(APP_NAME)_debug *.class

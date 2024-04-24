CC=gcc
CFLAGS=-Wall -Werror -std=c17
DEPS=utils.o layer.o

test_mnist: test_mnist.c $(DEPS)
	$(CC) -o $@ $< $(DEPS) $(CFLAGS)

.PHONY: clean

clean:
	rm $(wildcard *.o) $(wildcard *.exe)

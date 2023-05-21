all: clean build

default: build

############### FILES #################
FILES=src cy_env.pxd cy_jobgenfile.pxd cy_jobgensim.pxd setup.py pyenv.pyx pyjobgenfile.pyx pyjobgensim.pyx pylogreader.pyx

############## COMMANDS ###############
build: ${FILES}
	python3 setup.py build_ext -i

clean:
	rm -rf build
	rm -f *.cpython*.so pyenv.cpp pyjobgenfile.cpp pyjobgensim.cpp pylogreader.cpp
#######################################
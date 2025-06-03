SHELL:=/bin/sh
SHELLOPTS:=pipefail:errexit

UNAME=$(shell uname -s)

minimal: docker_built

deps_built:
	bash docker/install_deps.sh
	touch $@

docker_built: deps_built
	bash docker/install_docker.sh
	touch $@

vpf_built:
	cd third_party/VideoProcessingFramework && mkdir -p build && cd build && \
  cmake -D VIDEO_CODEC_SDK_INCLUDE_DIR=../../../third_party/Video_Codec_SDK/Interface \
        -D AVFORMAT_INCLUDE_DIR=$(FFMPEG)/include \
        -D AVUTIL_INCLUDE_DIR=$(FFMPEG)/include \
        -D AVCODEC_INCLUDE_DIR=$(FFMPEG)/include \
        -D FFMPEG_DIR=$(FFMPEG) \
        -D VIDEO_CODEC_SDK_DIR:PATH=$(VIDEO_CODEC_SDK) \
        -D NVCUVID_LIBRARY=../../../third_party/Video_Codec_SDK/Lib/linux/stubs/x86_64/libnvcuvid.so \
        -D NVENCODE_LIBRARY=../../../third_party/Video_Codec_SDK/Lib/linux/stubs/x86_64/libnvidia-encode.so \
        -D CMAKE_INSTALL_PREFIX=../../dist \
        -D GENERATE_PYTORCH_EXTENSION=on \
				-D PYTHON_EXECUTABLE=$(PYTHON_BINARY) \
        -D GENERATE_PYTHON_BINDINGS=ON .. && \
  make -j && make install && \
	cd .. && rm -rf build

linter_python: $(shell find . -type f -name "*.py" -not -path "*third_party*" -not -path "*venv*")
	black $?

# TODO: unused, remove?
update_pythonpath:
	export PYTHONPATH=$PYTHONPATH:`pwd`

# TODO: unused, remove?
poetry_shell:
	cd $(APP_PATH)
	poetry shell

# TODO: unused, remove?
test_built:
	$(PYTHON_BINARY) -m unittest discover tests/
	touch $@

build_env:
	cd $(APP_PATH) && poetry install && cd -

build_env_extras:
	cd $(APP_PATH) && poetry install --extras $(EXTRAS) && cd -

activate_env:
	cd $(APP_PATH) && . `poetry env info --path`/bin/activate && cd -

protos/python/intercom/%_pb2_grpc.py protos/python/intercom/%_pb2.py: idl/intercom/proto/%.proto
	mkdir -p protos/python/intercom
	$(PYTHON_BINARY) -m grpc_tools.protoc -Iidl/intercom --python_out=protos/python/intercom --grpc_python_out=protos/python/intercom proto/$*.proto

protos/python/intercom/%_pb2.py protos/python/intercom/%_pb2_grpc.py: idl/intercom/proto/%.proto
	mkdir -p $(dir $@)
	$(PYTHON_BINARY) -m grpc_tools.protoc -Iidl/intercom --python_out=protos/python/intercom --grpc_python_out=protos/python/intercom proto/$*.proto

.PHONY: pb_py_built
pb_py_built: protos/python/intercom/sensor_pb2.py protos/python/intercom/sensor_pb2_grpc.py

clean:
	rm -rf *_built
	rm -rf *.egg-info
	rm -rf idl
	rm -rf third_party/dist

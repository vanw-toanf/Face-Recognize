import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRT_Engine:
    """Lớp Wrapper để tải và thực thi model TensorRT."""
    def __init__(self, engine_path, max_batch_size=1):
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for binding in self.engine:
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def __call__(self, host_input: np.ndarray):
        batch_size = host_input.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Kích thước batch ({batch_size}) lớn hơn max_batch_size ({self.max_batch_size})")

        host_input = np.ascontiguousarray(host_input)
        cuda.memcpy_htod_async(self.inputs[0]['device'], host_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        output_shape = tuple([batch_size] + list(self.outputs[0]['shape'][1:]))
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
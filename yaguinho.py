import cupy as cp

class CupySumBenchmark:
  def __init__(self, size):
    self.size = size
    self.data = None
    self.result = None
    self.gpu_time = None

  def run(self):
    # Eventos para medir tempo de GPU
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()

    # Iniciar gravação de tempo GPU
    start_event.record()

    # Gerar vetor aleatório na GPU
    self.data = cp.random.rand(self.size)

    # Executar soma na GPU
    self.result = cp.sum(self.data)

    # Finalizar gravação de tempo GPU
    stop_event.record()
    # Sincronizar para garantir que a medida seja correta
    stop_event.synchronize()

    # Cálculo do tempo de GPU em milissegundos
    self.gpu_time = cp.cuda.get_elapsed_time(start_event, stop_event)

    # Trazer o resultado para a CPU e imprimir
    final_result = self.result.get()
    print(f"Resultado da soma: {final_result}")
    print(f"Tempo total GPU (ms): {self.gpu_time:.3f}")

if __name__ == "__main__":
  tamanho = 100000001
  benchmark = CupySumBenchmark(tamanho)
  benchmark.run()
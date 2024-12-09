import threading
import numpy as np
import time

class Threads_yaguinho:
    def dividir_lista_em_listas(self, lista, num_partes=2):
        """
        Divide a lista em 'num_partes' sublistas.
        """
        tamanho_sublista = len(lista) // num_partes
        return [lista[i * tamanho_sublista:(i + 1) * tamanho_sublista] for i in range(num_partes)] + [lista[num_partes * tamanho_sublista:]]

    def processar_threads(self, lista, num_threads=2):
        """
        Processa a soma da lista usando 'num_threads' threads.
        """
        sublistas = self.dividir_lista_em_listas(lista, num_threads)
        resultados = [0] * num_threads
        threads = []

        def thread_funcao_principal(sub_lista, indice):
            # Usando NumPy para liberar o GIL durante a soma
            resultados[indice] = np.sum(sub_lista)

        for i in range(num_threads):
            thread = threading.Thread(target=thread_funcao_principal, args=(sublistas[i], i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return sum(resultados)

    def calcular_serial(self, lista):
        """
        Calcula a soma da lista na forma serial.
        """
        return np.sum(lista)  # Usando NumPy para eficiência


# Função para medir o tempo de execução
def medir_tempo_execucao(func, *args, **kwargs):
    inicio = time.time()
    resultado = func(*args, **kwargs)
    fim = time.time()
    return resultado, fim - inicio


# Lista de exemplo para o teste
minha_lista = np.arange(1, 100000001)  # Lista com 100.000.000 elementos

yaguinho = Threads_yaguinho()

# Serial
resultado_serial, tempo_serial = medir_tempo_execucao(yaguinho.calcular_serial, minha_lista)
print(f"Serial: Resultado={resultado_serial}, Tempo={tempo_serial:.6f} segundos")

# 2 threads
resultado_2_threads, tempo_2_threads = medir_tempo_execucao(yaguinho.processar_threads, minha_lista, 2)
print(f"2 Threads: Resultado={resultado_2_threads}, Tempo={tempo_2_threads:.6f} segundos")

# 4 threads
resultado_4_threads, tempo_4_threads = medir_tempo_execucao(yaguinho.processar_threads, minha_lista, 4)
print(f"4 Threads: Resultado={resultado_4_threads}, Tempo={tempo_4_threads:.6f} segundos")

# 8 threads
resultado_8_threads, tempo_8_threads = medir_tempo_execucao(yaguinho.processar_threads, minha_lista, 8)
print(f"8 Threads: Resultado={resultado_8_threads}, Tempo={tempo_8_threads:.6f} segundos")

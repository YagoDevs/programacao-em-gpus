#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

/*
Antes, o código estava dividindo a lista em pares e ímpares.
Isso estava adicionando um overhead desnecessário, já que duas listas
precisavam ser criadas, processadas e somadas separadamente.

Agora, o código calcula diretamente a soma da lista completa,
tornando o processo mais eficiente e reduzindo o tempo de paralelização.
*/

// Função que inicializa uma lista com números inteiros aleatórios
void inicializar_lista(std::vector<int>& lista, int tamanho) {
    srand(static_cast<unsigned>(time(nullptr)));  // Inicializa a semente para números aleatórios
    for (int i = 0; i < tamanho; i++) {
        lista.push_back(rand() % 1000);  // Gera números aleatórios entre 0 e 999
    }
}

// Função para calcular a soma de maneira paralela
/*
Antes: O código dividia a lista em pares e ímpares, calculava a soma de cada lista separadamente
em duas seções paralelas, e depois combinava os resultados. Isso criava overhead adicional
e dificultava o escalonamento e agora a soma é feita diretamente sobre a lista completa usando `#pragma omp parallel for` com
uma redução, eliminando a necessidade de separar pares e ímpares.
*/
long long soma_paralela(const std::vector<int>& lista, int num_threads) {
    long long soma_total = 0;

    // Define o número de threads permitido
    omp_set_num_threads(num_threads);

    // Paralelizando o cálculo da soma
    #pragma omp parallel for reduction(+:soma_total)
    for (size_t i = 0; i < lista.size(); i++) {
        soma_total += lista[i];
    }

    return soma_total;
}

// Função para calcular a soma de maneira serial
long long soma_serial(const std::vector<int>& lista) {
    long long soma = 0;
    for (int valor : lista) {
        soma += valor;
    }
    return soma;
}

int main() {
    /*
    Antes: O código principal estava configurado para processar uma lista dividida em pares e ímpares.
    Isso criava duas chamadas de funções diferentes para calcular as somas e verificar os resultados e agora o código principal foi simplificado. Ele processa a soma diretamente na lista original,
    eliminando o trabalho desnecessário e testando a soma paralela com diferentes números de threads.
    */
    int tamanho = 100000000;  // Tamanho grande da lista para testar desempenho
    std::vector<int> lista;
    inicializar_lista(lista, tamanho);

    // Soma serial
    double inicio_serial = omp_get_wtime();
    long long resultado_serial = soma_serial(lista);
    double tempo_serial = omp_get_wtime() - inicio_serial;
    std::cout << "Soma Serial: " << resultado_serial << ", Tempo Serial: " << tempo_serial << " segundos" << std::endl;

    // Soma paralelizada com diferentes números de threads
    for (int num_threads = 2; num_threads <= 8; num_threads *= 2) {
        double inicio_paralelo = omp_get_wtime();
        long long resultado_paralelo = soma_paralela(lista, num_threads);
        double tempo_paralelo = omp_get_wtime() - inicio_paralelo;

        // Verifica se os resultados coincidem
        if (resultado_serial != resultado_paralelo) {
            std::cerr << "Erro: Resultados não coincidem!" << std::endl;
            return -1;
        }

        std::cout << "Soma Paralela com " << num_threads << " threads: " << resultado_paralelo
                  << ", Tempo Paralelo: " << tempo_paralelo << " segundos" << std::endl;
    }

    return 0;
}

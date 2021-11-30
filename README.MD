# RTRGAN - Representativeness Tuning GAN

## Termo de Ajuste de Representatividade (TAR) para a função de custo (Representativeness Tuning Rate)

De acordo com os metodos citados na introdução para correção do **Fairness** e mitigação de **bias** em algoritimos de aprendizagem de maquina, muito metodos existentes e em uso focam em uma abordagem mais reativas, normalmente resolvendo um problema de otimização das métricas já conhecidas para medir o nivel de justica nos dados, como o **demographic parity, disparate impact, equalized odd and equalized opportunities**.
No entanto acreditamos que uma oportunidade menos explorada e com muito potencial para solução do fairness nas bases de dados é a modelagem analitica da origem do vies, conforme citado no trabalho \cite{pessach2020algorithmic}, como **missing data, historical bias ou proxying attributes**, definir uma função que descreva numericamente o efeito desse fenomeno social que dá origem ao vies no algoritmo de aprendizado de maquina, e embutir esse termo na função de custo para mitigar o efeito desse fenomeno.

## definição matematica do TAR
Assumindo que a representatividade da população alvo, definida nesse trabalho como a população detentora dos atributos sensíveis diretamente afetados pelo vies, nos dados gerados $D(\hat{x},\hat{s},\hat{y})$, pode ser definida em um função $R_u$, definida como $R_u(\hat{x},\hat{s},\hat{y}) = \sum\limits_{i=m}^{m}\bbone_{s=1}$, e a representatividade da população, não alvo, que vamos chamar nesse trabalho de população previlegiada, é definida por uma função $R_p$, que é $R_p(\hat{x},\hat{s},\hat{y}) = \sum\limits_{i=m}^{m}\bbone_{s=0}$.
\par A métrica TAR \emph{(taxa de ajuste de representatividade)}, tem o intuito de minimizar a distância da representatividade no dataset entre os grupos $R_u$ e $R_p$, isso é feito pela seguinte expressão.

\begin{equation}
    \nu\EX\left[\frac{\lvert R_p(\hat{x},\hat{s},\hat{y}-R_u(\hat{x},\hat{s},\hat{y}) \rvert}{m}\right]
\end{equation}
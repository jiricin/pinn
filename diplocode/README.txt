Složka "diplocode":

V kódu ve spustitelné časti (__main__) je na začátku uvedena buď proměnná "func" nebo proměnné "mat_L" a "mat_R", které řídí konfiguraci úlohy. "func" může být změněna na hodnoty 1, 2, nebo 3 a odpovídá dané konfiguraci úlohy v tabulce 4.1. "mat_L" a "mat_R" zastupují konstanty tepelné vodivosti v tabulce 4.2.

Po spuštění bude neuronová síť trénovat 100 iteračních cyklů. Po dokončení této rutiny se do konzole vypíší detaily ztrátových funkcí. Zadáním nového počtu iterací do konzole bude neuronová síť dále trénovat. Pro dokončení procesu učení (po vykonání všech iterací) napíšeme do konzole 'q', kód poté vytvoří grafy chyb a průběhů ztrátových funkcí a program se ukončí.
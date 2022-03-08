@echo off
SET folder = %1
SET /A batchsize = %2
SET /A overlap = %3
SET /A size = %4
SET outfolder = %5
SET /A twoBatchsize = %batchsize% *2
SET /A rows_left = %size%
SET /A rows_gone = 0
SET /A rows_gone2 = 0
:while
SET /A rows_gone = %rows_gone2% + %batchsize%
SET /A rows_left = %size% - %rows_gone%
if %rows_left% GEQ %batchsize% (
    echo ./main.py %folder % %rows_gone2% %rows_gone% %outfolder %
    start cmd /k main.py %folder % %rows_gone2% %rows_gone% %outfolder %
    SET /A rows_gone2 = %rows_gone% - %overlap%
    goto :while
)
echo ./main.py %folder % %rows_gone2% %size% %outfolder %
start cmd /k main.py %folder % %rows_gone2% %size% %outfolder %
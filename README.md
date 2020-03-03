# DOC TYPE DEFINER

## USAGE

**usage:** usage: scanner.py [-h] [--file [PDF_FILE]] [--dir [DIR]] 
                  [--templates_file [TEMPLATES_FILE]]
                  [--templates_dir [TEMPLATES_DIR]] [--train] 
                  [--not_save]

**optional arguments:**
  -h, --help - show this help message and exit
  
  --file [PDF_FILE] - scan pdf file
  
  --dir [DIR] - scan directory with pdf files
  
  --templates_file [TEMPLATES_FILE] - file with templates. 'base.yml' - default
  
  --templates_dir [TEMPLATES_DIR] - directory with templates
  
  --train - train mode
  
  --not_save - save not training

## EXAMPLES

```./scanner.py --image tests/Паспорт\ РФ_2.pdf --train --not_save ```

```./scanner.py --dir tests/ --train ```

```./scanner.py --dir tests/ --train --templates_dir templates/ ```


## TEMPLATE FILE EXAMPLE

**from base.yml**:
```
Акт услуг:
  koef: 0.7
  акт: 1
  заказчик: 1
  исполнитель: 1
Паспорт РФ:
  koef: 0.3
  выдачи: 1
  гор.: 1
  дата: 1
  жен.: 1
  код: 1
  муж.: 1
  паспорт: 1
  подразделение: 1
  пол: 1
  российская: 1
  фамилия: 1
  федерация: 1
Счет:
  koef: 1
  итого: 1
  счет: 1
```
  
## TEMPLATES DIR
  
from Паспорт РФ.yml:
```
  koef: 0.3
  выдачи: 1
  гор.: 1
  дата: 1
  жен.: 1
  код: 1
  муж.: 1
  паспорт: 1
  подразделение: 1
  пол: 1
  российская: 1
  фамилия: 1
  федерация: 1
```
from Счет.yml:
```
  koef: 1
  итого: 1
  счет: 1
```

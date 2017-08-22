pyinstaller -F -i malefemale.ico -p c:\Anaconda3\Lib\site-packages\PyQt5\Qt\bin coupleswapper_gui.py -y
pyinstaller -F -c coupleswapper.py -n coupleswapper_console -i malefemale.ico -y
copy readme.txt .\dist

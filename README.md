- Clone repository
```
!git clone https://github.com/npk7264/Neural-style-transfer.git
```
- Move to folder Neural-style-transfer & install packages in requirements.txt. If you use Google Colab, you don't need to do this step
```
!pip install requirements.txt
```
- Run notebook.ipynb & config arg
```
!python main.py --content='assets/content.png' --style='assets/vangogh.jpg' --style_weight=1e9 --total_step 2000
```

import pandas as pd
import scipy.stats as st
df = pd.read_csv('CleanData2.csv').drop('Unnamed: 0',axis=1)
y = df['critical_temp']
y_now = y.apply(lambda x: (x ** 0.12 - 1) / 0.12)
ans = st.normaltest(y_now)
print(ans)
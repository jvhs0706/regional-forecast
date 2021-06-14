import os
from datetime import date

begin_dt = date(2015,1,1)
end_dt = date(2020,12,31)
tz = -1
datadir = '../../data'

def _verify_file(fn: str, begin_date: date, end_date: date, time_zone: int):
    with open(fn, 'r') as f:
        begin_date_str, end_date_str = begin_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
        l1 = f.readline()
        print(f'Verifying: {l1[12:-2]}')
        l2 = f.readline()
        assert f'Time(UTC{time_zone:+}): {begin_date_str}-{end_date_str}' == l2[1:-2], f'Wrong signature in {fn}!'

if __name__ == '__main__':
    for fn in os.listdir(datadir):
        _verify_file(f'{datadir}/{fn}', begin_dt, end_dt, tz)
    print('All data signature tests passed!')
    
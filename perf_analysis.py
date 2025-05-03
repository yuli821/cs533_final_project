
file_list = [
    '/work/hdd/beih/jkang8/a40_large_16/no_opt_large16.out',
    '/work/hdd/beih/jkang8/a40_large_16/opt_large16.out'
]

for file in file_list:
    usage_dict = {}
    with open(file, 'r') as f:
        print(f'collecting power consumption for {file}')
        for line in f:
            if 'power usage=' in line:
                device = line.split(' ')[1].split(':')[1]
                usage = float(line.split(' ')[7])
                if device in usage_dict:
                    usage_dict[device] += usage
                else:
                    usage_dict[device] = usage  
    total = 0
    for k, v in usage_dict.items():
        total += v
        print(f'power usage for gpu {k}: {v:.3f} Watts')
    print(f'total power usage: {total:.3f} Watts \n')
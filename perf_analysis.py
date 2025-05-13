import re

file_list = [
    'training.out'
]
total = 0
for file in file_list:
    usage_dict = {}
    with open(file, 'r') as f:
        print(f'collecting power consumption for {file}')
        for line in f:
            # if 'power usage=' in line:
                # print(line)
            match = re.search(r'power usage=\s*([0-9.]+)', line)
            if match:
                total += float(match.group(1))  
    # total = 0
    # for k, v in usage_dict.items():
        # total += v
        # print(f'power usage for gpu {k}: {v:.3f} Watts')
    print(f'total power usage: {total:.3f} Watts \n')
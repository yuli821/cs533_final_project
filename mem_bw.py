import re
import pandas as pd

def parse_gpumemsizesum(report_text):
    pattern = re.compile(
        r'^\s*([\d,]+\.\d+)\s+'      # Total (MB)
        r'([\d,]+)\s+'              # Count
        r'([\d\.]+)\s+'             # Avg (MB)
        r'([\d\.]+)\s+'             # Med (MB)
        r'([\d\.]+)\s+'             # Min (MB)
        r'([\d\.]+)\s+'             # Max (MB)
        r'([\d\.]+)\s+'             # StdDev (MB)
        r'(.*)$'                    # Operation
    )

    data = []
    in_table = False
    for line in report_text.splitlines():
        # Detect start of table by dashes
        if line.strip().startswith('-----------'):
            in_table = True
            continue
        if not in_table:
            continue
        m = pattern.match(line)
        if m:
            total_mb = float(m.group(1).replace(',', ''))
            count = int(m.group(2).replace(',', ''))
            avg_mb = float(m.group(3))
            med_mb = float(m.group(4))
            min_mb = float(m.group(5))
            max_mb = float(m.group(6))
            stddev_mb = float(m.group(7))
            operation = m.group(8).strip()
            data.append({
                'Operation': operation,
                'Total (MB)': total_mb,
                'Count': count,
                'Avg (MB)': avg_mb,
                'Med (MB)': med_mb,
                'Min (MB)': min_mb,
                'Max (MB)': max_mb,
                'StdDev (MB)': stddev_mb
            })

    return pd.DataFrame(data)

def parse_gpumemtime(report_text: str) -> pd.DataFrame:
    """
    Parse the 'gpumemtime' stats report section.
    Returns a DataFrame with columns:
      - Operation
      - Time (%) 
      - Total Time (ns)
      - Count
      - Avg (ns)
      - Med (ns)
      - Min (ns)
      - Max (ns)
      - StdDev (ns)
    """
    # Regex that allows commas in any integer or float field
    pat = re.compile(
        r'^\s*'                                 # leading space
        r'(\d+(?:\.\d+)?)\s+'                   # Time (%)
        r'([\d,]+)\s+'                          # Total Time (ns)
        r'([\d,]+)\s+'                          # Count
        r'([\d,]+(?:\.\d+)?)\s+'                # Avg (ns)
        r'([\d,]+(?:\.\d+)?)\s+'                # Med (ns)
        r'([\d,]+(?:\.\d+)?)\s+'                # Min (ns)
        r'([\d,]+(?:\.\d+)?)\s+'                # Max (ns)
        r'([\d,]+(?:\.\d+)?)\s+'                # StdDev (ns)
        r'(.*)$'                                # Operation
    )

    lines = report_text.splitlines()
    data = []
    in_table = False

    for line in lines:
        if not in_table:
            # detect header
            if 'Time (%)' in line and 'Operation' in line:
                in_table = True
            continue

        # skip separator
        if line.strip().startswith('---'):
            continue

        # stop at blank line
        if line.strip() == '':
            break

        m = pat.match(line)
        if not m:
            continue

        time_pct    = float(m.group(1))
        total_ns    = int(m.group(2).replace(',', ''))
        count       = int(m.group(3).replace(',', ''))
        avg_ns      = float(m.group(4).replace(',', ''))
        med_ns      = float(m.group(5).replace(',', ''))
        min_ns      = int(m.group(6).replace(',', ''))
        max_ns      = int(m.group(7).replace(',', ''))
        stddev_ns   = float(m.group(8).replace(',', ''))
        operation   = m.group(9).strip()

        data.append({
            'Operation':       operation,
            'Time (%)':        time_pct,
            'Total Time (ns)': total_ns,
            'Count':           count,
            'Avg (ns)':        avg_ns,
            'Med (ns)':        med_ns,
            'Min (ns)':        min_ns,
            'Max (ns)':        max_ns,
            'StdDev (ns)':     stddev_ns,
        })

    return pd.DataFrame(data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Parse gpumemsizesum stats report'
    )
    parser.add_argument('report_file', help='Path to the stats report')
    args = parser.parse_args()

    # Read the entire report
    with open(args.report_file, 'r') as f:
        text = f.read()

    df_time = parse_gpumemtime(text)
    print(df_time.to_markdown(index=False))
    df_size = parse_gpumemsizesum(text)
    print(df_size.to_markdown(index=False))
    # Merge size and time by Operation
    df = pd.merge(df_size, df_time, on='Operation', how='inner')

    # Compute Memory Bandwidth: MB/s = Total(MB) / (Total Time (ns) / 1e9)
    df['Mem_bw (MB/s)'] = df['Total (MB)'] * 1e9 / df['Total Time (ns)']

    # Only display Operation and Mem_bw
    result = df[['Operation', 'Mem_bw (MB/s)']]
    print(result.to_markdown(index=False))

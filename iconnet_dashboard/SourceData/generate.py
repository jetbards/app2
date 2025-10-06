import numpy as np
import pandas as pd
import os


def generate_sample_data(n_customers=1000, random_state=None):
    """Generate realistic sample data for ICONNET customers"""

    # Atur seed
    if random_state is not None:
        np.random.seed(random_state)  # reproducible kalau di-set
    else:
        np.random.seed()  # random tiap run

    # Segments dan distribusi
    segments = ['Residential', 'Corporate', 'Enterprise', 'Government']
    segment_probs = [0.65, 0.20, 0.10, 0.05]

    # Data dasar
    data = {
        'customer_id': range(1, n_customers + 1),
        'segment': np.random.choice(segments, n_customers, p=segment_probs),
        'tenure': np.random.randint(1, 72, n_customers),
        'contract_duration': np.random.choice(
            ['Monthly', '1 Year', '2 Years'],
            n_customers,
            p=[0.4, 0.4, 0.2]
        ),
    }

    # Inisialisasi array tambahan
    internet_speed = np.zeros(n_customers)
    monthly_charges = np.zeros(n_customers)
    service_type = []
    additional_services = []

    # Generate data berdasarkan segment
    for i in range(n_customers):
        segment = data['segment'][i]

        if segment == 'Residential':
            # Residential internet packages
            speeds = [35, 50, 100]
            speed_probs = [0.4, 0.4, 0.2]
            speed = np.random.choice(speeds, p=speed_probs)

            # Base pricing based on speed (from the product PDF)
            base_price = 265000 if speed == 35 else (331000 if speed == 50 else 442000)

            internet_speed[i] = speed
            monthly_charges[i] = base_price * (0.9 if data['contract_duration'][i] != 'Monthly' else 1.0)
            service_type.append('Broadband')
            additional_services.append(np.random.choice(['None', 'IPTV', 'Static_IP'], p=[0.6, 0.3, 0.1]))

        elif segment == 'Corporate':
            # Corporate services
            services = ['Metro_Ethernet', 'IP_VPN', 'Clear_Channel']
            service = np.random.choice(services, p=[0.5, 0.3, 0.2])

            internet_speed[i] = np.random.choice([100, 200, 500, 1000])
            base_price = 10000000 if service == 'Metro_Ethernet' else (
                3000000 if service == 'IP_VPN' else 2000000
            )

            monthly_charges[i] = base_price * (0.85 if data['contract_duration'][i] == '2 Years' else 1.0)
            service_type.append(service)
            additional_services.append(np.random.choice(['Backup_Line', 'Premium_Support', 'None'], p=[0.4, 0.3, 0.3]))

        elif segment == 'Enterprise':
            # Enterprise managed services
            services = ['Managed_Office', 'Managed_Router', 'SD-WAN', 'Colocation']
            service = np.random.choice(services)

            internet_speed[i] = np.random.choice([500, 1000, 2000])
            base_price = 15000000 if service == 'Colocation' else 8000000
            monthly_charges[i] = base_price
            service_type.append(service)
            additional_services.append('Premium_Support')

        else:  # Government
            internet_speed[i] = np.random.choice([100, 200, 500])
            monthly_charges[i] = np.random.uniform(5000000, 15000000)
            service_type.append('Special_Government')
            additional_services.append('High_Security')
            

    # Tambahkan ke dataframe
    
    data.update({
        'internet_speed_mbps': internet_speed,
        'monthly_charges': monthly_charges,
        'service_type': service_type,
        'additional_services': additional_services,
        'monthly_usage_gb': np.random.lognormal(5, 1.2, n_customers),  # GB usage
        'downtime_minutes': np.random.exponential(120, n_customers),  # Monthly downtime
        'customer_satisfaction': np.random.randint(6, 11, n_customers),  # 6-10 scale
        'payment_method': np.random.choice(['Bank_Transfer', 'Credit_Card', 'Direct_Debit', 'Invoice'], n_customers),
        'complaint_count': np.random.poisson(1.5, n_customers),  # Number of complaints
        'payment_delay_days': np.random.exponential(5, n_customers),  # Payment delay in days
    })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Default folder output
    output_dir = r"D:\Skripsi\Code\app\SourceData"
    os.makedirs(output_dir, exist_ok=True)

    # Input jumlah customer
    try:
        n_customers = int(input("Masukkan jumlah customer (default: 1000): ").strip() or 1000)
    except ValueError:
        n_customers = 1000

    # Input nama file
    filename = input("Masukkan nama file CSV (default: Data.csv): ").strip()
    if not filename:
        filename = "Data.csv"
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    # Path full
    output_file = os.path.join(output_dir, filename)

    # Generate dan simpan
    df = generate_sample_data(n_customers=n_customers)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n✅ Sample data ({n_customers} customers) saved to: {output_file}\n")
    print("Preview Data:")
    print(df.head())

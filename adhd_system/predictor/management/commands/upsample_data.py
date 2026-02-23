import pandas as pd
import os
from django.core.management.base import BaseCommand
from imblearn.over_sampling import SMOTE
from django.conf import settings

class Command(BaseCommand):
    help = 'Upsamples the CSV dataset using SMOTE to balance classes (increasing 0s)'

    def handle(self, *args, **kwargs):
        # 1. Use a Raw String (r'') to fix the "Invalid Argument" / escape sequence error
        csv_path = r'C:\1-MINI_PROJECT\adhd_code\ADHD_COPY.csv'
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"File not found at: {csv_path}"))
            return

        self.stdout.write("Reading CSV...")
        # Load data and drop rows with missing values (NaN) which crash SMOTE
        df = pd.read_csv(csv_path).dropna()

        # 2. Check if 'target' exists
        if 'ADHD' not in df.columns:
            self.stdout.write(self.style.ERROR("Error: Column 'target' not found in CSV."))
            return

        # 3. Pre-process: SMOTE only works with numbers. 
        # get_dummies converts text columns (like Gender) into numbers (0 and 1)
        df_numeric = pd.get_dummies(df)

        X = df_numeric.drop('ADHD', axis=1)
        y = df_numeric['ADHD']

        self.stdout.write(f"Original distribution: {y.value_counts().to_dict()}")

        # 4. Apply SMOTE 
        # This will detect that you have fewer 0s and create synthetic 0s to match the 1s
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"SMOTE Error: {e}"))
            return

        # 5. Recombine into a single DataFrame
        df_upsampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        
        # 6. Save the new version
        # This saves it in your Django project folder: C:\1-MINI_PROJECT\adhd_code\adhd_system\upsampled_dataset.csv
        output_path = os.path.join(settings.BASE_DIR, 'upsampled_dataset.csv')
        df_upsampled.to_csv(output_path, index=False)

        self.stdout.write(self.style.SUCCESS(f"Successfully saved balanced data to {output_path}"))
        self.stdout.write(f"New distribution: {y_resampled.value_counts().to_dict()}")
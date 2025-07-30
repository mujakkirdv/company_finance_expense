📌 Overview
A comprehensive financial management system built with Python and Streamlit for tracking earnings, expenses, and investments across multiple categories. Designed specifically for manufacturing businesses with support for detailed cost categorization.

✨ Features
📊 Core Functionalities
Multi-category tracking (Main Category, Sub-Category, Expense Type)

Automatic balance calculations (Opening/Closing balance)

Date-wise transaction recording with intelligent date handling

Payment mode tracking (Cash, Bank Transfer, Mobile Banking etc.)

📈 Reporting & Analytics
Monthly financial summaries

Category-wise expense breakdowns

Investment tracking

Trend analysis (Daily/Weekly/Monthly/Yearly)

Financial health indicators

🛠️ Data Management
Excel-based data storage (easy to use)

Data validation and cleaning

Backup and restore functionality

Duplicate entry detection

🚀 Installation
Clone the repository:

bash
git clone https://github.com/yourusername/financial-management-system.git
cd financial-management-system
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run main.py
📂 File Structure
text
financial-management-system/
├── main.py                 # Main application script
├── expenses_earnings.xlsx  # Default data file
├── backup/                 # Auto-generated backups
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading and validation
│   ├── calculations.py     # Financial calculations
│   └── reports.py          # Report generation
└── README.md               # This file
🔍 Data Structure
The system uses the following columns in Excel:

Column Name	Description	Example
date	Transaction date	2024-03-08
descriptions	Transaction description	"PP Poly and Rubber"
main_category	Primary category	"Production or Cost of Goods Sold (COGS)"
sub_category	Secondary category	"Packaging Materials"
expense_type	Specific expense type	"Office Supplies"
payment_mode	Payment method	"Cash"
openning_balance	Starting balance	-1540
earnings_amount	Income amount	0
invesment_amount	Investment amount	5160
expenses_amount	Expense amount	1540
closing_balance	Calculated closing balance	-1540
🧑‍💻 Usage Guide
Adding New Transactions
Navigate to Data Entry page

Fill in all required fields

Click "Save Entry"

System will automatically calculate balances

Generating Reports
Go to Dashboard for quick overview

Use Financial Reports for detailed analysis

Select date range and categories as needed

Click on charts for interactive exploration

Managing Data
Edit Data page for direct modifications

Backup/Restore for data safety

Data Quality checks for consistency

🛠️ Troubleshooting
Common Issues
Excel file not found: Ensure expenses_earnings.xlsx exists in project root

Negative balances: Verify transaction amounts and sequence

Duplicate entries: Use the Data Quality page to identify duplicates


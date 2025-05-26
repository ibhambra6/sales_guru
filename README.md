# Sales Guru - AI-Powered Sales Automation System

Sales Guru is a production-ready AI-powered sales automation system built with CrewAI that can analyze any CSV file containing lead data and generate comprehensive sales materials including lead qualification, prospect research, email outreach, and sales call preparation.

## Features

- **Universal CSV Support**: Analyze any CSV file containing lead/prospect data
- **Multi-Agent AI System**: Specialized agents for different sales tasks
- **Comprehensive Output**: Lead qualification, prospect research, email templates, and call prep
- **Production Ready**: Robust error handling, logging, and retry mechanisms
- **Flexible Input**: Command-line arguments and environment variables
- **Multiple Output Formats**: JSON and Markdown outputs

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sales_guru

# Install dependencies
pip install .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Set Up API Keys

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

### 3. Prepare Your CSV File

Sales Guru can work with any CSV file containing lead/prospect data. Common column names it recognizes include:

- `company_name`, `company`, `organization`
- `contact_name`, `name`, `full_name`, `first_name`, `last_name`
- `email`, `email_address`
- `phone`, `phone_number`
- `website`, `company_website`, `url`
- `industry`, `sector`
- `title`, `job_title`, `position`
- `location`, `city`, `state`, `country`

**Example CSV structure (see `example_leads.csv` in the repository):**

```csv
company_name,contact_name,email,phone,website,industry,title,location,company_size
Acme Corporation,John Smith,john.smith@acme.com,555-0123,acme.com,Technology,CEO,"New York, NY",100-500
TechStart Solutions,Jane Doe,jane.doe@techstart.com,555-0456,techstart.com,Software,CTO,"San Francisco, CA",10-50
```

### 4. Run Sales Guru

Sales Guru runs completely non-interactively. All parameters can be specified via command line arguments or environment variables.

#### Basic Usage

```bash
# Run with default CSV file (knowledge/leads.csv)
sales-guru

# Specify CSV file
sales-guru --csv your_leads.csv

# Specify all parameters
sales-guru --csv leads.csv --company "Your Company" --description "Your company description"

# Use different CSV file with environment variable
export SALES_GURU_CSV_PATH=/path/to/your/leads.csv
sales-guru
```

## Usage Examples

### Example 1: Using the Included Example File

```bash
# Run with the included example CSV file
sales-guru --csv example_leads.csv \
  --company "Your Company Name" \
  --description "Your company description"
```

### Example 2: Real Estate Leads

```bash
sales-guru --csv real_estate_leads.csv \
  --company "Premier Realty" \
  --description "Full-service real estate agency specializing in residential and commercial properties"
```

### Example 3: SaaS Prospects

```bash
sales-guru --csv saas_prospects.csv \
  --company "CloudTech Solutions" \
  --description "Cloud-based software solutions for enterprise resource planning"
```

### Example 4: Manufacturing Leads

```bash
sales-guru --csv manufacturing_leads.csv \
  --company "Industrial Equipment Co" \
  --description "Industrial equipment manufacturer and supplier"
```

### Example 5: Using Environment Variables

```bash
# Set your default CSV file
export SALES_GURU_CSV_PATH=/path/to/your/leads.csv

# Run with defaults (no prompts)
sales-guru

# Override company info
sales-guru --company "New Company" --description "New description"
```

## Command Line Options

```bash
sales-guru [command] [options]

Commands:
  run      Run the sales automation process (default)
  train    Train the AI models
  test     Test the system
  replay   Replay a specific task

Options:
  --csv FILE              Path to CSV file containing leads
  --company NAME          Your company name
  --description TEXT      Your company description
  --verbose, -v           Enable verbose logging
  --iterations N          Number of iterations for train/test
  --model MODEL           OpenAI model for testing
  --task-id ID            Task ID for replay
  --output FILE           Output file for training results
```

## CSV File Requirements

### Minimum Requirements

- Must be a valid CSV file with headers
- Should contain at least company names or contact information
- File must be readable and properly formatted

### Recommended Columns

While Sales Guru is flexible with column names, these are the most useful:

**Essential:**

- Company name (any variation)
- Contact name or email

**Highly Recommended:**

- Email address
- Phone number
- Website
- Industry/sector

**Optional but Useful:**

- Job title/position
- Location information
- Company size
- Revenue information
- Any additional context

### Example CSV Templates

#### Basic Template

```csv
company_name,contact_name,email,website
Acme Corp,John Smith,john@acme.com,acme.com
TechStart,Jane Doe,jane@techstart.com,techstart.com
```

#### Comprehensive Template

```csv
company_name,contact_name,email,phone,website,industry,title,location,company_size
Acme Corp,John Smith,john@acme.com,555-1234,acme.com,Technology,CEO,"New York, NY",50-100
TechStart,Jane Doe,jane@techstart.com,555-5678,techstart.com,Software,CTO,"San Francisco, CA",10-50
```

## Output Files

Sales Guru generates both JSON and Markdown outputs:

### JSON Outputs (for programmatic use)

- `lead_qualification.json` - Qualified leads with scores and reasoning
- `prospect_research.json` - Detailed research on each prospect
- `email_outreach.json` - Personalized email templates
- `sales_call_prep.json` - Call preparation materials

### Markdown Outputs (human-readable)

- `lead_qualification.md` - Formatted lead qualification report
- `prospect_research.md` - Research findings and insights
- `email_outreach.md` - Email templates and strategies
- `sales_call_prep.md` - Call scripts and talking points

## Advanced Usage

### Environment Variables

```bash
# Set default CSV path
export SALES_GURU_CSV_PATH=/path/to/your/default/leads.csv

# Set environment (affects logging)
export ENVIRONMENT=production

# Set log level
export LOG_LEVEL=DEBUG
```

### Training the Model

```bash
# Train with your specific data
sales-guru train --csv your_data.csv --iterations 5 --output training_results.json
```

### Testing

```bash
# Test the system
sales-guru test --csv test_data.csv --model gpt-4o --iterations 3
```

## Architecture

Sales Guru uses a multi-agent architecture with specialized AI agents:

1. **Lead Qualification Agent** - Scores and qualifies leads
2. **Prospect Research Agent** - Conducts detailed research
3. **Email Outreach Agent** - Creates personalized emails
4. **Sales Call Prep Agent** - Prepares call materials
5. **Supervisor Agent** - Coordinates the workflow
6. **Markdown Conversion Agent** - Formats outputs

## Error Handling

The system includes comprehensive error handling:

- **API Rate Limiting**: Automatic retry with exponential backoff
- **File Validation**: Validates CSV files before processing
- **Fallback Models**: Switches to backup models if primary fails
- **Task Completion Guarantees**: Ensures all tasks complete successfully

## Logging

Logs are written to:

- Console (with appropriate level based on environment)
- `logs/sales_guru.log` (rotating file handler in production)

## Troubleshooting

### Common Issues

1. **CSV File Not Found**

   ```bash
   # Check file path
   ls -la your_file.csv
   # Use absolute path
   sales-guru --csv /full/path/to/your_file.csv
   ```
2. **API Key Issues**

   ```bash
   # Verify API keys are set
   echo $OPENAI_API_KEY
   echo $GOOGLE_API_KEY
   echo $SERPER_API_KEY
   ```
3. **Rate Limiting**

   - The system automatically handles rate limits
   - For heavy usage, consider upgrading API plans
   - Use `--verbose` to see retry attempts
4. **CSV Format Issues**

   - Ensure proper CSV formatting
   - Check for special characters in headers
   - Verify file encoding (UTF-8 recommended)

### Getting Help

```bash
# Show help
sales-guru --help

# Show version info
sales-guru --version

# Run with verbose logging
sales-guru --verbose --csv your_file.csv
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For support, please:

1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with:
   - Your CSV file structure (anonymized)
   - Error messages
   - Steps to reproduce

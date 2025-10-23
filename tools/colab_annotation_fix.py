import argparse
import pandas as pd


def main():
	parser = argparse.ArgumentParser(description='Fix Colab-generated annotation CSV paths and optionally remove header')
	parser.add_argument('--input', '-i', default='annotations/synthetic_annotations.csv', help='Input CSV path')
	parser.add_argument('--output', '-o', default='annotations/synthetic_annotations_noheader.csv', help='Output CSV path')
	parser.add_argument('--remove-header', action='store_true', help='Remove a single header row (skips first row)')
	parser.add_argument('--skip-rows', type=int, default=0, help='Number of rows to skip at the start of the file (overrides --remove-header if >0)')
	args = parser.parse_args()

	input_csv = args.input
	output_csv = args.output

	# determine skiprows
	skiprows = 0
	if args.skip_rows and args.skip_rows > 0:
		skiprows = args.skip_rows
	elif args.remove_header:
		skiprows = 1

	# Read CSV and optionally skip rows. If header is skipped we also avoid using header names
	if skiprows > 0:
		df = pd.read_csv(input_csv, skiprows=skiprows, header=None)
	else:
		df = pd.read_csv(input_csv, header=None)

	# Replace Windows slashes and remove leading folder path on column 0 if present
	if 0 in df.columns:
		df[0] = df[0].astype(str).str.replace("\\", "/", regex=False)
		df[0] = df[0].str.replace(r"^sentences_data_synth/", "", regex=True)

	# Save cleaned version without header and index
	df.to_csv(output_csv, header=False, index=False)

	print("✅ Removed header/skip rows, fixed paths, and saved to", output_csv)


if __name__ == '__main__':
	main()
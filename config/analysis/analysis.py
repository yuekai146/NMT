import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Analysis modes')
    
    # Add argument for score mode
    parser_vocab = subparsers.add_parser(
            'vocab', help='Analyse active data vocabulary'
            )
    parser_vocab.add_argument('-d', '--directory', type=str, required=True,
            help='The active data directory'
            )
    

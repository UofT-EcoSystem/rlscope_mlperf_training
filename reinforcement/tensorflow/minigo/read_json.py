#!/usr/bin/env python
import argparse
import jsmin
import json
import sys
from os import environ as ENV

def main():
    parser = argparse.ArgumentParser("Read an attribute from a json file.")
    parser.add_argument('json_path',
                        help="Path to a .json file")
    parser.add_argument('--attr', required=True,
                        help="Attribute name whose value we should read")
    parser.add_argument('--default',
                        help="If attribute isn't found, print this instead")
    parser.add_argument('--allow-env',
                        action='store_true',
                        help="If attribute isn't found, print this instead")
    parser.add_argument('--no-newline', '-n', action='store_true',
                        help="Don't print a newline after the value")
    args = parser.parse_args()

    def print_value(value):
        sys.stdout.write(str(value))
        if not args.no_newline:
            sys.stdout.write("\n")

    with open(args.json_path, 'r') as f:
        js_str = jsmin.jsmin(f.read())
        js = json.loads(js_str)

    value = None
    if args.allow_env and args.attr in ENV:
        value = ENV[args.attr]
    elif args.attr not in js:
        if args.default is not None:
            value = args.default
        else:
            print("ERROR: couldn't find {attr} in {path}".format(
                attr=args.attr, path=args.json_path))
            sys.exit(1)
    else:
        value = js[args.attr]

    assert value is not None

    print_value(value)

if __name__ == '__main__':
    main()
from models import get_db
from pymongo import MongoClient
import logging
import argparse
import gridfs
import pymongo
import datetime

def is_valid_iso8601(time_str: str) -> bool:
    """Check if the given string is a valid ISO 8601 datetime."""
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Delete rainfall and/or state GridFS files.")

    parser.add_argument('-s', '--start', type=str, required=True,
                        help='Start time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name of domain [AKL]')
    parser.add_argument('-p', '--product', type=str, required=True,
                        help='Name of product to delete [QPE, auckprec, qpesim]') 
    parser.add_argument('-c', '--cascade', default=False, action='store_true',
                        help='Delete the cascade files')
    parser.add_argument('-r', '--rain', default=False, action='store_true',
                        help='Delete the rainfall files')
    parser.add_argument('--params', default=False, action='store_true',
                        help='Delete the parameter documents')
    
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help='Only list files that would be deleted, don’t delete them.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not (args.rain or args.cascade or args.params):
        logging.warning("Nothing to delete: specify --rain, --cascade, or --params")
        return

    # Validate and parse times
    def parse_time(time_str):
        if not is_valid_iso8601(time_str):
            logging.error(f"Invalid time format: {time_str}")
            exit(1)
        t = datetime.datetime.fromisoformat(time_str)
        return t.replace(tzinfo=datetime.timezone.utc) if t.tzinfo is None else t

    start_time = parse_time(args.start)
    end_time = parse_time(args.end)

    name = args.name
    product = args.product
    dry_run = args.dry_run

    if product not in ["QPE", "auckprec", "qpesim", "nwpblend"]:
        logging.error(f"Invalid product: {product}")
        return

    db = get_db()

    def delete_files(collection_name):
        coll = db[f"{collection_name}.files"]
        fs = gridfs.GridFS(db, collection=collection_name) 

        if product == "QPE":
            query = {
                "metadata.product": product,
                "metadata.valid_time": {"$gte": start_time, "$lte": end_time}
            }
        else:
            query = {
                "metadata.product": product,
                "metadata.base_time": {"$gte": start_time, "$lte": end_time}
            }

        ids = list(coll.find(query, {"_id": 1,"filename":1}))
        count = len(ids)

        if dry_run:
            logging.info(f"[Dry Run] {count} files matched in {collection_name}. Listing _id values:")
            for doc in ids:
                logging.info(f"  Would delete: {doc['filename']}")
        else:
            for doc in ids:
                fs.delete(doc["_id"])
            logging.info(f"Deleted {count} files from {collection_name}")

    if args.rain:
        delete_files(f"{name}.rain")

    if args.cascade:
        delete_files(f"{name}.state")

    if args.params:
        collection_name = f"{name}.params"
        coll = db[collection_name]
        if product == "QPE":
            query = {
                "metadata.product": product,
                "metadata.valid_time": {"$gte": start_time, "$lte": end_time}
            }
        else:
            query = {
                "metadata.product": product,
                "metadata.base_time": {"$gte": start_time, "$lte": end_time}
            }

        ids = list(coll.find(query, {"_id": 1}))
        count = len(ids)
        if dry_run: 
            logging.info(f"[Dry Run] {count} files matched in {collection_name}")
        else:
            coll.delete_many(query)
            logging.info(f"Deleted {count} files from {collection_name}")


if __name__ == "__main__":
    main()

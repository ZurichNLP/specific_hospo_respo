#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive baseline approach that outputs a 'generic' response based on the input reivew rating.

Example call:
    python rule_based_method.py --rating_file ../data/hotel/500k/test.rating --outfile ../models/rule_based/translations.txt
"""

import argparse
import random
from collections import defaultdict

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rating_file", type=str, required=True, help="pathing to input rating file")
    ap.add_argument("--outfile", type=str, required=True, help="path to output file")
    return ap.parse_args()    


class RuleBasedResponder:


    def __init__(self, seed=42):
        self.SEED = seed
        self.database = defaultdict(set)
    
    def update(self, rating, response):        
        self.database[rating].add(response)

    def respond(self, rating):
        random.seed(self.SEED)
        return random.sample(self.database[rating], 1)[0]

if __name__ == '__main__':
    args = set_args()

    data = [
        (5, "Thank you for taking the time to write a review. We are very happy to hear you enjoyed your stay and we look forward to welcoming you back soon. Kind regards"),
        (5, "Many thanks for your review! We are pleased you enjoyed your stay with us. We truly appreciate our guests taking time out in their busy lives to write a review. Thanks once again and looking forward to seeing you again. Best wishes"),
        (4, "Thanks for staying with us. We appreciate your feedback!"),
        (4, "Hello and thank you for your review! We are happy to hear that you enjoyed your stay with us. We hope to see you again soon! Sincerely"),
        (3, "Thank you for your valuable review. We always want to provide the best experience to our guests and I am sorry to hear that on this occassion we did not succeed. We hope you will give us the chance to make it up to you on your next visit and we look forward to welcoming you again. Best regards"),
        (3, "Thank you so much for posting a review of our hotel. We are constantly seeking ways to develop and deliver an exceptional stay for our guests and hearing valuable feedback from our them is paramount. We hope to address the bulk of your concerns quickly. Thanks again for your review. Sincerely"),
        (2, "Thank you for taking the time to write a review. May it be constructive or positive criticism; we value all guests' opinions and feedback. We want to thank you for choosing to stay with us and sincerely apologize for any inconvenience. Hopefully you will give us another opportunity to show an improvement next time you stay with us. Sincerely"),
        (2, "Thank you for submitting your review of our Cambridge Orchard Park Travelodge. We're so sorry to hear about your recent experience and would like to hear more about your stay. May we kindly request you contact us via our website with your review so our customer service team can investigate? Thank you again for posting your comments and we hope to hear from you soon"),
        (1, "Hello, thank you for your opinion and feedback. I am disappointed and I apologize that your stay with us was less than pleasant. If you have the time, our operations manager, would like to hear from you in regards to these issues so we may better the stay for future guests. Thank you again for expressing your concerns."),
        (1, "Thank you for taking the time to write your review. It disappoints me to read some of the things which you have expressed. We generally pride ourselves on maintaining high standards and providing optimum service to our guests. It seems on this occasion we did not succeed and we are so sorry to have disappointed you. Your comments do raise a few concerns for me that I would like to look into a little further. It would be great if you could get in touch with me so that we can discuss some of the points in more detail so that corrective action can be taken. Please be assured that we do take these matters very seriously and I hope to hear from you soon. Once again our sincere apologies. Warm Regards")
        ]

    responder = RuleBasedResponder()

    for rate, resp in data:
        responder.update(rate, resp)

    total_ratings = len(responder.database)
    total_responses = sum(len(v) for v in list(responder.database.values()))
    print(f"built responder from data with {total_ratings} ratings and {total_responses} unqiue responses...")

    with open(args.rating_file, 'r', encoding='utf8') as ratings:
        with open(args.outfile, 'w', encoding='utf8') as outf:
            for line in ratings:
                rating = int(line.strip()) 
                outf.write(f'{responder.respond(rating)}\n')
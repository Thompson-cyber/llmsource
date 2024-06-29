from scapy.layers.dns import DNS
from layers.properties.dns_dir import DNSDir
from layers.properties.dns_opcode import DNSOpCode
from layers.properties.dns_qtype import DNSQType
from layers.properties.dns_rcode import DNSRCode
from layers.dns import DNS as myDNS


class DNSParser:

    @classmethod
    def parse_dns(cls, dns_layer: DNS) -> tuple[myDNS, int, int]:
        if dns_layer.qd:
            qtype = DNSQType(dns_layer.qd.qtype)
            qname = dns_layer.qd.qname
        else:
            qtype = None
            qname = None
        if DNSDir(dns_layer.qr) == DNSDir.RESPONSE:
            answers = cls.parse_dns_answers(dns_layer)
            rcode = DNSRCode(dns_layer.rcode)
            if not answers:
                answers = None
        else:
            answers = None
            rcode = None
        return myDNS(
            dns_layer.id,
            DNSDir(dns_layer.qr), DNSOpCode(dns_layer.opcode),
            qtype,
            rcode,
            qname,
            answers
        ), len(dns_layer), len(dns_layer.payload)

    @staticmethod
    def parse_dns_answers(dns_layer) -> list[dict[str, str | int]]:
        """Generated with ChatGPT.

        Parse DNS answers from a Scapy DNS layer object to extract answer fields and their values in
        a dictionary.

        :param dns_layer: Scapy DNS response object
        :return: dictionary containing answer fields and their values
        """
        answers = []
        answer_rr = dns_layer.an  # Start with the first answer in the response
        while answer_rr:
            answers.append(answer_rr.fields.copy())
            answer_rr = answer_rr.payload  # Move to the next answer, if any
            if not answer_rr:  # Stop if no more answers
                break
        return answers

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# You need to run this script under python2.x
import urllib
import re
import Queue

base = 'https://giantpanda.gtisc.gatech.edu/malrec/'
treated_list = []
untreat_list = []


def url_is_in_site(url):
    if True:
        return True
    else:
        return False


def get_form(url):
    html = urllib.urlopen(url).read()
    # print html
    print "*******************************"
    reg = r'<form.*\r.*form>'
    form_list = re.findall(reg, html, re.S)
    with open('form.txt', 'w') as f:
        f.write(form_list[0])
    print 'OK'

    # iframe情况

    # js writeForm情况，iframe+js writeForm

dest_dir="./malrec_pcaps/"

def downLoadPicFromURL(url,jpg_url):
    try:
        urllib.urlretrieve(url, dest_dir+jpg_url)
    except Exception as e:
        print '\tError retrieving the URL:', url, e

def get_jpg(url):
    html = urllib.urlopen(url).read()
    reg = r'a href=[\'\"](.*.pcap)[\'\"]'
    jpg_list = re.findall(reg, html)
    jpg_list = list(set(jpg_list))
    return jpg_list


uq = Queue.Queue(maxsize=-1)
uq.put(base + '/pcap/')

jpg_count = 0
down_jpg = []
while not uq.empty():
    print uq.qsize()
    url = uq.get()
    print url
    print '=================================================================='

    jpg_list = get_jpg(url)


    for jpg_url in jpg_list:
        if jpg_url[:4] != 'http':
            real_url = base +'pcap/'+ jpg_url
            if jpg_url not in down_jpg:
                print real_url
                jpg_count += 1
                try:
                    downLoadPicFromURL(real_url,jpg_url)
                    down_jpg.append(jpg_url)
                # except IOError:
                #    print "url wrong!!!"
                except Exception:
                    print "url wrong!!!"
        else:
            print '[wrong jpg url]: %s' % jpg_url

    print "get list done!"


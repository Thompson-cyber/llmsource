#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0
#
# Copyright (c) 2013-2023 Igor Pecovnik, igor@armbian.com
#
# This file is a part of the Armbian Build Framework
# https://github.com/armbian/build/
#

# Disclaimer: this script was written solely using GitHub Copilot.
# I wrote "prompt" comments and the whole thing was generated by Copilot.
# Unfortunately I removed most original comments/prompts after code was generated, I should have kept them all in...
# I'm not sure if I should be proud or ashamed of this. <-- this was suggested by Copilot too.
# -- rpardini, 23/11/2022

import hashlib
import logging
import os

import common.aggregation_utils as util
import common.armbian_utils as armbian_utils
from common.md_asset_log import SummarizedMarkdownWriter

# Prepare logging
armbian_utils.setup_logging()
log: logging.Logger = logging.getLogger("aggregation")

# Read SRC from the environment, treat it.
armbian_build_directory = armbian_utils.get_from_env_or_bomb("SRC")
if not os.path.isdir(armbian_build_directory):
	raise Exception("SRC is not a directory")

# OUTPUT from the environment, treat it.
output_file = armbian_utils.get_from_env_or_bomb("OUTPUT")
with open(output_file, "w") as bash:
	bash.write("")

BUILD_DESKTOP = armbian_utils.yes_or_no_or_bomb(armbian_utils.get_from_env_or_bomb("BUILD_DESKTOP"))
BUILD_MINIMAL = armbian_utils.yes_or_no_or_bomb(armbian_utils.get_from_env_or_bomb("BUILD_MINIMAL"))
INCLUDE_EXTERNAL_PACKAGES = True
ARCH = armbian_utils.get_from_env_or_bomb("ARCH")
DESKTOP_ENVIRONMENT = armbian_utils.get_from_env("DESKTOP_ENVIRONMENT")
DESKTOP_ENVIRONMENT_CONFIG_NAME = armbian_utils.get_from_env("DESKTOP_ENVIRONMENT_CONFIG_NAME")
RELEASE = armbian_utils.get_from_env_or_bomb("RELEASE")  # "kinetic"
USERPATCHES_PATH = armbian_utils.get_from_env_or_bomb("USERPATCHES_PATH")

# Show the environment
armbian_utils.show_incoming_environment()

util.SELECTED_CONFIGURATION = armbian_utils.get_from_env_or_bomb("SELECTED_CONFIGURATION")  # "cli_standard"
util.DESKTOP_APPGROUPS_SELECTED = armbian_utils.parse_env_for_tokens("DESKTOP_APPGROUPS_SELECTED")  # ["browsers", "chat"]
util.SRC = armbian_build_directory

util.AGGREGATION_SEARCH_ROOT_ABSOLUTE_DIRS = [
	f"{armbian_build_directory}/config",
	f"{armbian_build_directory}/config/optional/_any_board/_config",
	f"{armbian_build_directory}/config/optional/architectures/{ARCH}/_config",
	f"{USERPATCHES_PATH}"
]

util.DEBOOTSTRAP_SEARCH_RELATIVE_DIRS = ["cli/_all_distributions/debootstrap", f"cli/{RELEASE}/debootstrap"]
util.CLI_SEARCH_RELATIVE_DIRS = ["cli/_all_distributions/main", f"cli/{RELEASE}/main"]

util.DESKTOP_ENVIRONMENTS_SEARCH_RELATIVE_DIRS = [
	f"desktop/_all_distributions/environments/_all_environments",
	f"desktop/_all_distributions/environments/{DESKTOP_ENVIRONMENT}",
	f"desktop/_all_distributions/environments/{DESKTOP_ENVIRONMENT}/{DESKTOP_ENVIRONMENT_CONFIG_NAME}",
	f"desktop/{RELEASE}/environments/_all_environments",
	f"desktop/{RELEASE}/environments/{DESKTOP_ENVIRONMENT}",
	f"desktop/{RELEASE}/environments/{DESKTOP_ENVIRONMENT}/{DESKTOP_ENVIRONMENT_CONFIG_NAME}"]

util.DESKTOP_APPGROUPS_SEARCH_RELATIVE_DIRS = [
	f"desktop/_all_distributions/appgroups",
	f"desktop/_all_distributions/environments/{DESKTOP_ENVIRONMENT}/appgroups",
	f"desktop/{RELEASE}/appgroups",
	f"desktop/{RELEASE}/environments/{DESKTOP_ENVIRONMENT}/appgroups"]

# Debootstrap.
debootstrap_packages = util.aggregate_all_debootstrap("packages")
debootstrap_packages_remove = util.aggregate_all_debootstrap("packages.remove")

# both main and additional result in the same thing, just different filenames.
rootfs_packages_main = util.aggregate_all_cli("packages")
rootfs_packages_external = util.aggregate_all_cli("packages.external")  # @TODO: enable/disable this
rootfs_packages_all = rootfs_packages_main
rootfs_packages_all = util.merge_lists(rootfs_packages_all, rootfs_packages_external, "add")
rootfs_packages_remove = util.aggregate_all_cli("packages.remove")
if not BUILD_MINIMAL:
	rootfs_packages_additional = util.aggregate_all_cli("packages.additional")
	rootfs_packages_all = util.merge_lists(rootfs_packages_all, rootfs_packages_additional, "add")

# Desktop environment packages; packages + packages.external
desktop_packages_main = util.aggregate_all_desktop("packages")
desktop_packages_external = util.aggregate_all_desktop("packages.external")
desktop_packages_additional = util.aggregate_all_desktop("packages.additional")
desktop_packages_all = util.merge_lists(desktop_packages_main, desktop_packages_external, "add")
desktop_packages_all = util.merge_lists(desktop_packages_all, desktop_packages_additional, "add")
desktop_packages_remove = util.aggregate_all_desktop("packages.remove")

env_list_remove = util.parse_env_for_list("REMOVE_PACKAGES")
env_list_extra_rootfs = util.parse_env_for_list("EXTRA_PACKAGES_ROOTFS")
env_list_extra_image = util.parse_env_for_list("EXTRA_PACKAGES_IMAGE")
env_package_list_board = util.parse_env_for_list(
	"PACKAGE_LIST_BOARD", {"function": "board", "path": "board.conf", "line": 0})
env_package_list_family = util.parse_env_for_list(
	"PACKAGE_LIST_FAMILY", {"function": "family", "path": "family.conf", "line": 0})
env_package_list_board_remove = util.parse_env_for_list(
	"PACKAGE_LIST_BOARD_REMOVE", {"function": "board_remove", "path": "board.conf", "line": 0})
env_package_list_family_remove = util.parse_env_for_list(
	"PACKAGE_LIST_FAMILY_REMOVE", {"function": "family_remove", "path": "family.conf", "line": 0})

# Now calculate the final lists.

# debootstrap is the aggregated list, minus the packages we want to remove.
AGGREGATED_PACKAGES_DEBOOTSTRAP = util.merge_lists(debootstrap_packages, debootstrap_packages_remove, "remove")
AGGREGATED_PACKAGES_DEBOOTSTRAP = util.merge_lists(AGGREGATED_PACKAGES_DEBOOTSTRAP, env_list_remove, "remove")

# components for debootstrap is just the aggregated list; or is it?
AGGREGATED_DEBOOTSTRAP_COMPONENTS = util.aggregate_all_debootstrap("components")
AGGREGATED_DEBOOTSTRAP_COMPONENTS_COMMA = ','.join(AGGREGATED_DEBOOTSTRAP_COMPONENTS).replace(' ', ',')

# The rootfs list; add the extras, and remove the removals.
AGGREGATED_PACKAGES_ROOTFS = util.merge_lists(rootfs_packages_all, env_list_extra_rootfs, "add")
AGGREGATED_PACKAGES_ROOTFS = util.merge_lists(AGGREGATED_PACKAGES_ROOTFS, rootfs_packages_remove, "remove")
AGGREGATED_PACKAGES_ROOTFS = util.merge_lists(AGGREGATED_PACKAGES_ROOTFS, env_package_list_board_remove, "remove")
AGGREGATED_PACKAGES_ROOTFS = util.merge_lists(AGGREGATED_PACKAGES_ROOTFS, env_package_list_family_remove, "remove")
AGGREGATED_PACKAGES_ROOTFS = util.merge_lists(AGGREGATED_PACKAGES_ROOTFS, env_list_remove, "remove")

# The desktop list.
AGGREGATED_PACKAGES_DESKTOP = util.merge_lists(desktop_packages_all, desktop_packages_remove, "remove")
AGGREGATED_PACKAGES_DESKTOP = util.merge_lists(AGGREGATED_PACKAGES_DESKTOP, env_package_list_board_remove, "remove")
AGGREGATED_PACKAGES_DESKTOP = util.merge_lists(AGGREGATED_PACKAGES_DESKTOP, env_package_list_family_remove, "remove")
AGGREGATED_PACKAGES_DESKTOP = util.merge_lists(AGGREGATED_PACKAGES_DESKTOP, env_list_remove, "remove")

# the image list; this comes from env only; apply the removals.
AGGREGATED_PACKAGES_IMAGE = util.merge_lists(env_list_extra_image, env_package_list_board, "add")
AGGREGATED_PACKAGES_IMAGE = util.merge_lists(AGGREGATED_PACKAGES_IMAGE, env_package_list_family, "add")
AGGREGATED_PACKAGES_IMAGE = util.merge_lists(AGGREGATED_PACKAGES_IMAGE, env_package_list_board_remove, "remove")
AGGREGATED_PACKAGES_IMAGE = util.merge_lists(AGGREGATED_PACKAGES_IMAGE, env_package_list_family_remove, "remove")
AGGREGATED_PACKAGES_IMAGE = util.merge_lists(AGGREGATED_PACKAGES_IMAGE, env_list_remove, "remove")

# Calculate a md5 hash of the list of packages, so we can use it as a cache key.
# This has to reflect perfectly what is done in create-cache.sh::create_new_rootfs_cache()
all_packages_in_cache = []
all_packages_in_cache.extend(util.only_names_not_removed(AGGREGATED_PACKAGES_DEBOOTSTRAP))
all_packages_in_cache.extend(util.only_names_not_removed(AGGREGATED_PACKAGES_ROOTFS))
all_packages_in_cache.extend(util.only_names_not_removed(AGGREGATED_PACKAGES_DESKTOP))
all_packages_in_cache_unique_sorted = sorted(set(all_packages_in_cache))
# @TODO: remove the package.uninstalls? (debsums case? also some gnome stuff)

AGGREGATED_ROOTFS_HASH_TEXT = "\n".join([f"pkg: {pkg}" for pkg in all_packages_in_cache_unique_sorted])
# @TODO: if apt sources changed, the hash should change too; add them (and their gpg keys too?) to the hash text with "apt: " prefix.
log.debug(f"<AGGREGATED_ROOTFS_HASH_TEXT>\n{AGGREGATED_ROOTFS_HASH_TEXT}\n</AGGREGATED_ROOTFS_HASH_TEXT>")

AGGREGATED_ROOTFS_HASH = hashlib.md5(AGGREGATED_ROOTFS_HASH_TEXT.encode("utf-8")).hexdigest()

# We need to aggregate some desktop stuff, which are not package lists, postinst contents and such.
# For this case just find the potentials, and for each found, take the whole contents and join via newlines.
AGGREGATED_DESKTOP_POSTINST = util.aggregate_all_desktop(
	"debian/postinst", util.aggregate_simple_contents_potential)
AGGREGATED_DESKTOP_CREATE_DESKTOP_PACKAGE = util.aggregate_all_desktop(
	"armbian/create_desktop_package.sh", util.aggregate_simple_contents_potential)
AGGREGATED_DESKTOP_BSP_POSTINST = util.aggregate_all_desktop(
	"debian/armbian-bsp-desktop/postinst", util.aggregate_simple_contents_potential)
AGGREGATED_DESKTOP_BSP_PREPARE = util.aggregate_all_desktop(
	"debian/armbian-bsp-desktop/prepare.sh", util.aggregate_simple_contents_potential)

# Aggregate the apt-sources; only done if BUILD_DESKTOP is True, otherwise empty.
AGGREGATED_APT_SOURCES = {}
if BUILD_DESKTOP:
	apt_sources_debootstrap = util.aggregate_all_debootstrap("sources/apt", util.aggregate_apt_sources)
	apt_sources_cli = util.aggregate_all_cli("sources/apt", util.aggregate_apt_sources)
	apt_sources_desktop = util.aggregate_all_desktop("sources/apt", util.aggregate_apt_sources)
	AGGREGATED_APT_SOURCES = util.merge_lists(apt_sources_debootstrap, apt_sources_cli, "add")
	AGGREGATED_APT_SOURCES = util.merge_lists(AGGREGATED_APT_SOURCES, apt_sources_desktop, "add")

# ----------------------------------------------------------------------------------------------------------------------


output_lists: list[tuple[str, str, object, object]] = [
	("debootstrap", "AGGREGATED_PACKAGES_DEBOOTSTRAP", AGGREGATED_PACKAGES_DEBOOTSTRAP, None),
	("rootfs", "AGGREGATED_PACKAGES_ROOTFS", AGGREGATED_PACKAGES_ROOTFS, None),
	("image", "AGGREGATED_PACKAGES_IMAGE", AGGREGATED_PACKAGES_IMAGE, None),
	("desktop", "AGGREGATED_PACKAGES_DESKTOP", AGGREGATED_PACKAGES_DESKTOP, None),
	("apt-sources", "AGGREGATED_APT_SOURCES", AGGREGATED_APT_SOURCES, util.encode_source_base_path_extra)
]

with open(output_file, "w") as bash, SummarizedMarkdownWriter("aggregation.md", "Aggregation") as md:
	bash.write("#!/bin/env bash\n")

	# loop over the aggregated lists
	for id, name, value, extra_func in output_lists:
		stats = util.prepare_bash_output_array_for_list(bash, md, name, value, extra_func)
		md.add_summary(f"{id}: {stats['number_items']}")

	# extra: if DESKTOP, add number of DESKTOP_APPGROUPS_SELECTED to the summary
	if BUILD_DESKTOP:
		md.add_summary(f"desktop_appgroups: {len(util.DESKTOP_APPGROUPS_SELECTED)}")

	# The rootfs hash (md5) is used as a cache key.
	bash.write(f"declare -g -r AGGREGATED_ROOTFS_HASH='{AGGREGATED_ROOTFS_HASH}'\n")  # (this done simply cos it has no newlines)
	bash.write(util.bash_string_multiline("AGGREGATED_ROOTFS_HASH_TEXT", AGGREGATED_ROOTFS_HASH_TEXT))
	# add_summary with the first 16 chars of the hash @TODO: unify the cropping of the hash vs bash
	md.add_summary(f"hash: {AGGREGATED_ROOTFS_HASH[:16]}")

	# Special case for components: debootstrap also wants a list of components, comma separated.
	bash.write(
		f"declare -g -r AGGREGATED_DEBOOTSTRAP_COMPONENTS_COMMA='{AGGREGATED_DEBOOTSTRAP_COMPONENTS_COMMA}'\n")

	# Single string stuff for desktop packages postinst's and preparation. @TODO use functions instead of eval.
	bash.write(util.prepare_bash_output_single_string(
		"AGGREGATED_DESKTOP_POSTINST", AGGREGATED_DESKTOP_POSTINST))
	bash.write(util.prepare_bash_output_single_string(
		"AGGREGATED_DESKTOP_CREATE_DESKTOP_PACKAGE", AGGREGATED_DESKTOP_CREATE_DESKTOP_PACKAGE))
	bash.write(util.prepare_bash_output_single_string(
		"AGGREGATED_DESKTOP_BSP_POSTINST", AGGREGATED_DESKTOP_BSP_POSTINST))
	bash.write(util.prepare_bash_output_single_string(
		"AGGREGATED_DESKTOP_BSP_PREPARE", AGGREGATED_DESKTOP_BSP_PREPARE))
	bash.write("\n## End of aggregation output\n");

	# 2) @TODO: Some removals... uninstall-inside-cache and such. (debsums case? also some gnome stuff)

	#    aggregate_all_cli "packages.uninstall" " "
	#    aggregate_all_desktop "packages.uninstall" " "
	#    PACKAGE_LIST_UNINSTALL="$(cleanup_list aggregated_content)"
	#    unset aggregated_content

	# Debug potential paths:
	all_potentials = util.get_all_potential_paths_packages()
	md.write(f"## Potential paths \n")
	for potential in all_potentials:
		md.write(f"- `{potential}`\n")

	log.debug(f"Done. Output written to {output_file}")
